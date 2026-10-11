#!/usr/bin/env python
"""
Linear-probe lung segmentation with a frozen ViT-B (e.g. microsoft/rad-dino)
on JSRT/SCR, Montgomery and Shenzhen.

Pipeline (same idea as the RAD-DINO / DINOv2 linear head):
  1. Load images + lung masks from the three datasets (binary: left+right lung).
  2. Run the frozen encoder ONCE and cache the patch tokens to disk.
  3. Train BatchNorm + 1x1 conv on the cached features (fast, works on CPU too).
  4. Report Dice per dataset on a held-out test split.

Multiple seeds:
  --seeds 0 1 2        runs the whole probe once per seed (default: 0 1 2) and reports
                       mean +/- std across seeds. Features are extracted only once; each
                       seed only re-trains the linear heads.
                       By default each seed also draws its own train/val/test split.
  --split_seed 0       keep the SAME split for every seed (only head init / shuffling varies).
  --verbose            show training progress (per-epoch loss / val Dice) and extraction logs.

Install:
  pip install torch transformers pillow numpy timm   (timm only needed for --checkpoint)

Own .pth backbone (DINOv2 / MAE / MoCo-v3 / timm-style ViT weights):
  python lung_linear_probe.py --checkpoint my_vitb.pth --img_size 518 ...
  (--img_size must be a multiple of the patch size: 518 for patch 14, 512 for patch 16)

Two .pth files (e.g. split-learning backbone without patch embedding + original DINO weights):
  python lung_linear_probe.py --checkpoint split_backbone.pth --base_checkpoint dino_vitb.pth ...
  Weights in --checkpoint win; --base_checkpoint fills everything --checkpoint does not contain.

Layers: by default a separate linear head is trained on the LAST block (-1) and on the
SECOND-LAST block (-2), and both are reported. Options:
  --layers -1 -2 --layer_mode both      also adds a head on the two layers concatenated
  --layers -1 -2 -3 -4 --layer_mode concat   DINOv2 "linear4"-style probe

Expected data layout (point the flags at your folders; searches are recursive):
  --jsrt_dir        folder with JSRT *.IMG files (2048x2048 raw)
  --scr_dir         SCR masks folder, containing ".../left lung/*.gif" and ".../right lung/*.gif"
  --montgomery_dir  MontgomerySet folder (CXR_png/, ManualMask/leftMask/, ManualMask/rightMask/)
  --shenzhen_img_dir   Shenzhen CXR_png folder
  --shenzhen_mask_dir  Shenzhen lung masks (e.g. from the Kaggle "Chest Xray Masks and Labels"
                       set, named CHNCXR_0001_0_mask.png)
Any dataset you leave out is skipped.

Example:
  python lung_linear_probe.py --encoder microsoft/rad-dino \
      --jsrt_dir data/JSRT --scr_dir data/SCR \
      --montgomery_dir data/MontgomerySet \
      --shenzhen_img_dir data/ChinaSet/CXR_png --shenzhen_mask_dir data/ChinaSet/mask \
      --seeds 0 1 2
"""
import argparse
import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif")

VERBOSE = False


def vprint(*a, **k):
    """Print only with --verbose (training progress, extraction progress, cache messages)."""
    if VERBOSE:
        print(*a, **k)


# ----------------------------------------------------------------------------- loading
def load_gray(path):
    """Load an image as float32 in [0, 1]. Handles JSRT .IMG raw files and 8/16-bit images."""
    path = str(path)
    if path.lower().endswith(".img"):  # JSRT: 2048x2048, 16-bit big-endian, 12-bit data
        a = np.fromfile(path, dtype=">u2").reshape(2048, 2048).astype(np.float32)
    else:
        a = np.array(Image.open(path)).astype(np.float32)
        if a.ndim == 3:
            a = a[..., :3].mean(-1)
    lo, hi = np.percentile(a, [0.5, 99.5])
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def load_mask(paths):
    """Union of one or more mask files -> bool array."""
    m = None
    for p in paths:
        a = np.array(Image.open(p).convert("L")) > 0
        m = a if m is None else (m | a)
    return m


def resize(a, size, nearest=False):
    im = Image.fromarray(a.astype(np.float32), mode="F")
    im = im.resize((size, size), Image.NEAREST if nearest else Image.BILINEAR)
    return np.array(im)


def rglob(root, patterns):
    out = []
    for pat in patterns:
        out += glob.glob(os.path.join(root, "**", pat), recursive=True)
    return sorted(set(out))


def collect_jsrt(jsrt_dir, scr_dir):
    imgs = rglob(jsrt_dir, ["*.IMG", "*.img"])
    left = {Path(p).stem.upper(): p for p in rglob(scr_dir, ["*left lung*/*.gif"])}
    right = {Path(p).stem.upper(): p for p in rglob(scr_dir, ["*right lung*/*.gif"])}
    items = []
    for p in imgs:
        s = Path(p).stem.upper()
        if s in left and s in right:
            items.append((p, [left[s], right[s]]))
    return items


def collect_montgomery(root):
    imgs = rglob(root, ["CXR_png/*.png"])
    left = {Path(p).name: p for p in rglob(root, ["leftMask/*.png"])}
    right = {Path(p).name: p for p in rglob(root, ["rightMask/*.png"])}
    return [(p, [left[Path(p).name], right[Path(p).name]])
            for p in imgs if Path(p).name in left and Path(p).name in right]


def collect_shenzhen(img_dir, mask_dir):
    masks = {Path(p).stem: p for p in rglob(mask_dir, ["*.png"])}
    items = []
    for p in rglob(img_dir, ["*.png"]):
        s = Path(p).stem
        m = masks.get(s + "_mask") or masks.get(s)
        if m:
            items.append((p, [m]))
    return items


# ----------------------------------------------------------------------------- .pth backbone
PREFIXES = ("module.", "teacher.backbone.", "student.backbone.", "backbone.",
            "encoder.", "model.", "vit.", "base_encoder.", "momentum_encoder.")


EXTRA_PREFIXES = []


def load_state_dict(path, key=None):
    """Load a .pth file and return a flat ViT state dict with common wrapper prefixes removed."""
    prefixes = tuple(EXTRA_PREFIXES) + PREFIXES
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, nn.Module):  # whole pickled model
        ckpt = ckpt.state_dict()
    if key:
        ckpt = ckpt[key]
    else:  # DINOv2 / MAE / MoCo / Lightning style nesting
        for k in ("teacher", "model", "state_dict", "student", "model_ema"):
            if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
                print(f"  {Path(path).name}: using weights under key '{k}'")
                ckpt = ckpt[k]
                break
    # Flat dict holding both networks (teacher.backbone.* and student.backbone.*):
    # keep only the teacher, otherwise stripping prefixes would let one overwrite the other.
    if any(k.startswith("teacher.") for k in ckpt):
        ckpt = {k: v for k, v in ckpt.items() if k.startswith("teacher.")}
        print("  using TEACHER weights (flat checkpoint with teacher.* keys)")
    sd = {}
    for k, v in ckpt.items():
        if not torch.is_tensor(v):
            continue
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k, changed = k[len(p):], True
        sd[k] = v
    # drop heads (DINO/iBOT projection heads, classifier, MAE decoder)
    drop = ("head", "dino_head", "ibot_head", "fc.", "decoder", "mask_token")
    return {k: v for k, v in sd.items() if not k.startswith(drop)}


def fuse_qkv(sd):
    """Fuse separate q/k/v projections into timm's attn.qkv (weights and biases, order q, k, v).

    LoRA is assumed to be ALREADY merged into q_proj / v_proj; leftover lora_A / lora_B
    tensors are dropped.
    """
    sd = dict(sd)
    lora = [k for k in sd if k.endswith((".lora_A", ".lora_B"))]
    for k in lora:
        del sd[k]
    if lora:
        print(f"  dropped {len(lora)} leftover LoRA tensors (LoRA assumed already merged)")

    q_keys = [k for k in sd if k.endswith("attn.q_proj.weight")]
    for qk in q_keys:
        p = qk[: -len("q_proj.weight")]
        sd[p + "qkv.weight"] = torch.cat([sd.pop(p + f"{x}_proj.weight") for x in "qkv"], 0)
        if p + "q_proj.bias" in sd:
            sd[p + "qkv.bias"] = torch.cat([sd.pop(p + f"{x}_proj.bias") for x in "qkv"], 0)
    if q_keys:
        print(f"  fused separate q/k/v projections into attn.qkv for {len(q_keys)} blocks")
    return sd


def merge_state_dicts(primary, base):
    """Weights from `primary` win; `base` fills every key that `primary` lacks."""
    merged = dict(base)
    shape_clash = [k for k in primary if k in base and primary[k].shape != base[k].shape]
    merged.update(primary)
    from_primary = [k for k in primary if k in base]
    only_primary = [k for k in primary if k not in base]
    from_base = [k for k in base if k not in primary]
    print(f"merged checkpoints: {len(from_primary)} keys from --checkpoint override base, "
          f"{len(from_base)} keys filled from --base_checkpoint, "
          f"{len(only_primary)} keys only in --checkpoint")
    if from_base:
        print("  filled from base:", _summarise(from_base))
    if only_primary:
        print("  WARNING keys only in --checkpoint (names may not match the base model, "
              "check prefixes / block numbering):", _summarise(only_primary))
    if shape_clash:
        print("  WARNING shape mismatch, using --checkpoint version:", shape_clash)
    return merged


def _summarise(keys, n=12):
    """Group keys like blocks.3.attn.qkv.weight -> blocks.*.attn.qkv.weight for readable logs."""
    import re
    groups = sorted({re.sub(r"\.\d+\.", ".*.", k) for k in keys})
    return groups[:n] + (["..."] if len(groups) > n else [])


class TimmViTEncoder(nn.Module):
    """Wraps a timm VisionTransformer built to match the checkpoint's shapes."""

    def __init__(self, sd, img_size):
        super().__init__()
        import timm.models.vision_transformer as tv

        w = sd["patch_embed.proj.weight"]            # (C, 3, p, p)
        embed_dim, patch = w.shape[0], w.shape[-1]
        depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        num_heads = embed_dim // 64
        reg = sd.get("register_tokens", sd.get("reg_token"))
        n_reg = reg.shape[1] if reg is not None else 0
        layerscale = any(".ls1." in k for k in sd)
        mlp_ratio = sd["blocks.0.mlp.fc1.weight"].shape[0] / embed_dim
        has_cls = "cls_token" in sd

        self.vit = tv.VisionTransformer(
            img_size=img_size, patch_size=patch, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, num_classes=0, reg_tokens=n_reg,
            class_token=has_cls, init_values=1e-5 if layerscale else None,
            dynamic_img_size=True, global_pool="token" if has_cls else "avg",
        )
        # timm's filter handles DINOv2 renames (register_tokens -> reg_token) and pos_embed resizing
        sd = tv.checkpoint_filter_fn(sd, self.vit)
        missing, unexpected = self.vit.load_state_dict(sd, strict=False)
        print(f"loaded .pth: embed_dim={embed_dim} depth={depth} patch={patch} "
              f"registers={n_reg} layerscale={layerscale}")
        if missing:
            print("  MISSING keys (randomly initialised!):", missing)
        if unexpected:
            print("  unexpected keys (ignored):", unexpected)
        self.config = type("cfg", (), {"patch_size": patch})()

    def forward_grid(self, x, layers):
        """layers: negative block indices, e.g. [-1, -2] -> {-1: grid, -2: grid} (final norm applied)."""
        depth = len(self.vit.blocks)
        idx = [depth + l for l in layers]
        outs = self.vit.get_intermediate_layers(x, n=idx, reshape=True, norm=True)
        return dict(zip(layers, outs))


# ----------------------------------------------------------------------------- features
@torch.no_grad()
def extract(model, x, patch, layers):
    """Frozen encoder -> {layer: (B, C, g, g) patch grid}; layer -1 = last block, -2 = second last.
    CLS/register tokens are dropped; the final LayerNorm is applied to every layer (DINOv2 protocol)."""
    if isinstance(model, TimmViTEncoder):
        return model.forward_grid(x, layers)
    out = model(pixel_values=x, output_hidden_states=True)
    g = x.shape[-1] // patch
    n = g * g
    ln = getattr(model, "layernorm", None)
    grids = {}
    for l in layers:
        h = out.hidden_states[l]            # hidden_states[-1] = output of the last block
        if ln is not None:
            h = ln(h)
        t = h[:, -n:, :]
        grids[l] = t.transpose(1, 2).reshape(x.shape[0], -1, g, g)
    return grids


def build_cache(name, items, model, args, mean, std, patch, device):
    enc_name = Path(args.encoder).name
    if args.checkpoint:  # add a hash of the full paths: many runs save "teacher_checkpoint.pth"
        import hashlib
        key = (str(Path(args.checkpoint).resolve()) + "|" + str(Path(args.base_checkpoint).resolve() if args.base_checkpoint else "")
               + "|qkv-fused")
        enc_name = f"{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}_{hashlib.md5(key.encode()).hexdigest()[:8]}"
    layer_tag = "_".join(f"L{-l}" for l in args.layers)       # e.g. L1_L2
    tag = f"{enc_name}_{args.img_size}_{layer_tag}"
    cache = Path(args.cache_dir) / f"{name}_{tag}.pt"
    if cache.exists():
        vprint(f"[{name}] loading cached features {cache}")
        return torch.load(cache)

    feats = {l: [] for l in args.layers}
    masks = []
    batch = []

    def flush():
        x = torch.stack([b for b in batch]).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            grids = extract(model, x, patch, args.layers)
        for l, f in grids.items():
            feats[l].append(f.half().cpu())
        batch.clear()

    for i, (img_p, mask_ps) in enumerate(items):
        img = load_gray(img_p)
        m = load_mask(mask_ps)
        if m.shape != img.shape:  # e.g. SCR masks are 1024x1024, JSRT images 2048x2048
            m = np.array(Image.fromarray(m.astype(np.uint8) * 255)
                         .resize((img.shape[1], img.shape[0]), Image.NEAREST)) > 0
        # Lungs should be dark; if they're brighter than the rest the image is inverted (JSRT).
        if img[m].mean() > img[~m].mean():
            img = 1.0 - img
        x = torch.from_numpy(resize(img, args.img_size)).float()
        x = x[None].repeat(3, 1, 1)
        x = (x - mean[:, None, None]) / std[:, None, None]
        batch.append(x)
        masks.append(torch.from_numpy(resize(m, args.mask_size, nearest=True) > 0.5).to(torch.uint8))
        if len(batch) == args.extract_bs:
            flush()
        if (i + 1) % 50 == 0:
            vprint(f"[{name}] {i + 1}/{len(items)}")
    if batch:
        flush()

    data = {"feats": {l: torch.cat(v) for l, v in feats.items()}, "masks": torch.stack(masks)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache)
    vprint(f"[{name}] cached {len(items)} images -> {cache}")
    return data


# ----------------------------------------------------------------------------- head + training
class LinearHead(nn.Module):
    def __init__(self, c, n_classes=2):
        super().__init__()
        self.bn = nn.BatchNorm2d(c)
        self.conv = nn.Conv2d(c, n_classes, 1)

    def forward(self, x):
        return self.conv(self.bn(x))


def dice_per_image(pred, target, eps=1e-6):
    pred, target = pred.float().flatten(1), target.float().flatten(1)
    inter = (pred * target).sum(1)
    return (2 * inter + eps) / (pred.sum(1) + target.sum(1) + eps)


def loss_fn(logits, masks):
    ce = F.cross_entropy(logits, masks.long())
    prob = logits.softmax(1)[:, 1]
    soft_dice = dice_per_image(prob, masks).mean()
    return ce + (1 - soft_dice)


@torch.no_grad()
def predict(head, feats, mask_size, device, bs=32):
    head.eval()
    preds = []
    for i in range(0, len(feats), bs):
        f = feats[i:i + bs].to(device).float()
        logits = F.interpolate(head(f), size=mask_size, mode="bilinear", align_corners=False)
        preds.append(logits.argmax(1).cpu())
    return torch.cat(preds)


def split(n, seed, frac=(0.7, 0.15)):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def make_parts(cached, split_seed, args):
    """Per-dataset train/val/test split of the cached features; train/val pooled over datasets."""
    parts = {"train": [], "val": [], "test": {}}
    for name, d in cached.items():
        n = len(d["masks"])
        tr, va, te = split(n, split_seed, (args.train_frac, args.val_frac))

        def take(idx):  # feats is {layer: tensor}; slice every layer with the same indices
            idx = torch.as_tensor(idx, dtype=torch.long)
            return {l: f[idx] for l, f in d["feats"].items()}, d["masks"][idx]
        parts["train"].append(take(tr))
        parts["val"].append(take(va))
        parts["test"][name] = take(te)
    return parts


def select(feat_dict, layers):
    return torch.cat([feat_dict[l] for l in layers], dim=1)


def train_probe(parts, layers, seed, args, device):
    """Train one linear head; return (head, {"val": best val Dice, dataset: mean test Dice})."""
    tr_f = torch.cat([select(f, layers) for f, _ in parts["train"]])
    tr_m = torch.cat([m for _, m in parts["train"]])
    va_f = torch.cat([select(f, layers) for f, _ in parts["val"]])
    va_m = torch.cat([m for _, m in parts["val"]])
    vprint(f"train {len(tr_f)} | val {len(va_f)} | feature grid {tuple(tr_f.shape[1:])}")

    torch.manual_seed(seed)
    head = LinearHead(tr_f.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_state = -1, None
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(tr_f))
        tot = 0.0
        for i in range(0, len(perm), args.bs):
            b = perm[i:i + args.bs]
            f = tr_f[b].to(device).float()
            m = tr_m[b].to(device)
            logits = F.interpolate(head(f), size=args.mask_size, mode="bilinear", align_corners=False)
            loss = loss_fn(logits, m)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        val_dice = dice_per_image(predict(head, va_f, args.mask_size, device), va_m).mean().item()
        if val_dice > best:
            best, best_state = val_dice, {k: v.clone() for k, v in head.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            vprint(f"epoch {ep:3d} | loss {tot / len(tr_f):.4f} | val Dice {100 * val_dice:.2f}")

    head.load_state_dict(best_state)
    res = {"val": 100 * best}
    for name, (f, m) in parts["test"].items():
        d = dice_per_image(predict(head, select(f, layers), args.mask_size, device), m) * 100
        res[name] = d.mean().item()
    return head, res


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="microsoft/rad-dino", help="HF model id or local path")
    ap.add_argument("--checkpoint", help="path to a .pth ViT checkpoint (overrides --encoder)")
    ap.add_argument("--ckpt_key", help="key inside the .pth holding the weights, e.g. 'teacher'")
    ap.add_argument("--base_checkpoint",
                    help="second .pth (e.g. original DINO weights) that fills every weight "
                         "missing from --checkpoint (e.g. patch embedding in split learning)")
    ap.add_argument("--base_ckpt_key", help="key inside --base_checkpoint holding the weights")
    ap.add_argument("--strip_prefix", nargs="*", default=[],
                    help="extra key prefixes to remove, e.g. server_model. client.")
    ap.add_argument("--mean", type=float, nargs=3, help="input normalisation mean (for .pth)")
    ap.add_argument("--std", type=float, nargs=3, help="input normalisation std (for .pth)")
    ap.add_argument("--jsrt_dir"); ap.add_argument("--scr_dir")
    ap.add_argument("--montgomery_dir")
    ap.add_argument("--shenzhen_img_dir"); ap.add_argument("--shenzhen_mask_dir")
    ap.add_argument("--img_size", type=int, default=518, help="multiple of the patch size (14)")
    ap.add_argument("--mask_size", type=int, default=256, help="resolution for loss / Dice")
    ap.add_argument("--layers", type=int, nargs="+", default=[-1, -2],
                    help="blocks to probe: -1 = last, -2 = second last, ...")
    ap.add_argument("--layer_mode", choices=["separate", "concat", "both"], default="separate",
                    help="separate: one linear head per layer; concat: one head on the "
                         "concatenated layers; both: all of these")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--extract_bs", type=int, default=8)
    ap.add_argument("--cache_dir", default="feature_cache")
    ap.add_argument("--seeds", "--seed", dest="seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="one full run per seed; results reported as mean +/- std across seeds")
    ap.add_argument("--split_seed", type=int, default=None,
                    help="use this seed for the data split in every run (default: each run "
                         "uses its own seed for the split too)")
    ap.add_argument("--verbose", action="store_true",
                    help="print training progress and feature-extraction logs")
    ap.add_argument("--train_frac", type=float, default=0.7, help="fraction of each dataset for training")
    ap.add_argument("--val_frac", type=float, default=0.15,
                    help="fraction for validation; the rest (1 - train - val) is the test set")
    args = ap.parse_args()
    VERBOSE = args.verbose
    test_frac = 1 - args.train_frac - args.val_frac
    if args.train_frac <= 0 or args.val_frac <= 0 or test_frac <= 0:
        raise SystemExit("--train_frac and --val_frac must be > 0 and sum to less than 1")
    split_desc = (f"fixed split (seed {args.split_seed})" if args.split_seed is not None
                  else "split re-drawn per seed")
    print(f"split per dataset: train {args.train_frac:.0%} | val {args.val_frac:.0%} | "
          f"test {test_frac:.0%} | seeds {args.seeds} | {split_desc}")

    EXTRA_PREFIXES.extend(args.strip_prefix)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- datasets
    datasets = {}
    if args.jsrt_dir and args.scr_dir:
        datasets["JSRT"] = collect_jsrt(args.jsrt_dir, args.scr_dir)
    if args.montgomery_dir:
        datasets["Montgomery"] = collect_montgomery(args.montgomery_dir)
    if args.shenzhen_img_dir and args.shenzhen_mask_dir:
        datasets["Shenzhen"] = collect_shenzhen(args.shenzhen_img_dir, args.shenzhen_mask_dir)
    datasets = {k: v for k, v in datasets.items() if v}
    for k, v in datasets.items():
        print(f"{k}: {len(v)} image/mask pairs")
    if not datasets:
        raise SystemExit("No image/mask pairs found - check the paths.")

    # --- frozen encoder
    if args.checkpoint:
        sd = fuse_qkv(load_state_dict(args.checkpoint, args.ckpt_key))
        if args.base_checkpoint:
            base = fuse_qkv(load_state_dict(args.base_checkpoint, args.base_ckpt_key))
            sd = merge_state_dicts(sd, base)
        model = TimmViTEncoder(sd, args.img_size)
        mean = torch.tensor(args.mean or [0.485, 0.456, 0.406])
        std = torch.tensor(args.std or [0.229, 0.224, 0.225])
    else:
        model = AutoModel.from_pretrained(args.encoder)
        try:
            proc = AutoImageProcessor.from_pretrained(args.encoder)
            mean, std = torch.tensor(proc.image_mean), torch.tensor(proc.image_std)
        except Exception:
            mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    patch = getattr(model.config, "patch_size", 14)
    assert args.img_size % patch == 0, f"--img_size must be a multiple of {patch}"

    # --- cache features once (seed-independent)
    cached = {name: build_cache(name, items, model, args, mean, std, patch, device)
              for name, items in datasets.items()}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- which layer combinations to probe
    probes = []
    if args.layer_mode in ("separate", "both"):
        probes += [(l,) for l in args.layers]
    if args.layer_mode in ("concat", "both") and len(args.layers) > 1:
        probes.append(tuple(args.layers))
    nice = {-1: "last", -2: "second_last"}
    pnames = {layers: "+".join(nice.get(l, f"layer{l}") for l in layers) for layers in probes}

    names = list(cached.keys())
    cols = ["val"] + names

    # --- runs: one per seed
    print("\n===== Per-seed Dice (lungs), mean over test images =====")
    print(f"{'seed':>6s}  {'layers':22s}" + "".join(f"{c:>12s}" for c in cols))
    all_results = {pnames[l]: [] for l in probes}
    for seed in args.seeds:
        split_seed = seed if args.split_seed is None else args.split_seed
        parts = make_parts(cached, split_seed, args)
        for layers in probes:
            pname = pnames[layers]
            vprint(f"\n----- seed {seed}: linear probe on layer(s) {layers} ({pname}) -----")
            head, res = train_probe(parts, layers, seed, args, device)
            torch.save(head.state_dict(), f"linear_head_lungs_{pname}_seed{seed}.pt")
            all_results[pname].append(res)
            print(f"{seed:>6d}  {pname:22s}" + "".join(f"{res[c]:12.2f}" for c in cols))

    # --- summary over seeds
    print(f"\n===== Dice (lungs) over {len(args.seeds)} seeds: mean ± std across seeds =====")
    print(f"{'layers':22s}" + "".join(f"{c:>18s}" for c in cols))
    for pname, runs in all_results.items():
        row = f"{pname:22s}"
        for c in cols:
            v = np.array([r[c] for r in runs])
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            row += f"{f'{v.mean():.2f} ± {sd:.2f}':>18s}"
        print(row)
    print(f"\n(heads saved as linear_head_lungs_<layers>_seed<N>.pt; use --verbose for training logs)")


if __name__ == "__main__":
    main()