#!/usr/bin/env python
"""
Linear-probe segmentation on COVID-QU-Ex (Kaggle: anasmohammedtahir/covidqu) with a frozen ViT.

Put this file next to segmentation.py (the lung probe script); it reuses its backbone loading
(HF models like microsoft/rad-dino, or your own .pth + --base_checkpoint) and feature extraction.

Tasks
  --task infection   COVID infection masks ("Infection Segmentation Data"), default
  --task lung        lung masks ("Lung Segmentation Data", ~34k images; use --max_per_class)

Uses the OFFICIAL Train / Val / Test split of the dataset.

Multiple seeds
  --seeds 0 1 2      one linear head per seed (default 0 1 2); results are reported as
                     mean +/- std ACROSS SEEDS. Features are extracted once; the split is the
                     official one, so seeds vary head initialisation and batch order.
  --data_seed 0      seed for --max_per_class subsampling (kept fixed so the cache stays valid)
  --verbose          show training progress (per-epoch loss / val Dice) and extraction logs

Because many infection masks cover (almost) the whole lung, the infection task reports:
  * Dice over POSITIVE test images (COVID images with a non-empty infection mask)
  * Dice on FOCAL positives only (infection covers < --focal_max_coverage of the lung)
  * the LUNG-MASK BASELINE: Dice obtained by simply predicting the lung mask as "infection"
    -> the probe only localises infection if it beats this number
  * image-level detection (infection present?) from predicted area: AUROC, sens/spec

Example
  python src/covid_seg_probe.py --root /scratch-shared/svhonk/covidqu --task infection \
      --layers -1 -2 --layer_mode separate --seeds 0 1 2
  python src/covid_seg_probe.py --root ... --checkpoint split.pth --base_checkpoint dino.pth
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:  # the lung probe script, whatever you named it
    import utils.seg_evaluation.segmentation as base
except ImportError:
    import lung_linear_probe as base

from transformers import AutoImageProcessor, AutoModel

SPLITS = ("train", "val", "test")

VERBOSE = False


def vprint(*a, **k):
    """Print only with --verbose (training progress, extraction progress, cache messages)."""
    if VERBOSE:
        print(*a, **k)


# ----------------------------------------------------------------------------- dataset
def find_split_dirs(root, task):
    """Return {'train': Path, 'val': Path, 'test': Path} for the chosen task."""
    key = "infection" if task == "infection" else "lung segmentation"
    found = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_dir() and p.name.lower() in SPLITS and key in str(p).lower():
            found.setdefault(p.name.lower(), p)
    if len(found) < 3:  # maybe --root already points inside the task folder
        for s in SPLITS:
            for cand in Path(root).iterdir():
                if cand.is_dir() and cand.name.lower() == s:
                    found.setdefault(s, cand)
    missing = [s for s in SPLITS if s not in found]
    if missing:
        raise SystemExit(f"Could not find {missing} folders for task '{task}' under {root}")
    return found


def collect_split(split_dir, task, classes, max_per_class, seed):
    """Items: dicts with image path, target mask path, lung mask path, class name."""
    items = []
    for cdir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
        if classes and cdir.name not in classes:
            continue
        subs = {d.name.lower(): d for d in cdir.iterdir() if d.is_dir()}
        img_dir = subs.get("images")
        lung_dir = next((d for n, d in subs.items() if "lung" in n and "mask" in n), None)
        inf_dir = next((d for n, d in subs.items() if "infection" in n and "mask" in n), None)
        tgt_dir = inf_dir if task == "infection" else lung_dir
        if img_dir is None or tgt_dir is None:
            print(f"  skipping {cdir} (no images/ or mask folder found: {list(subs)})")
            continue
        cls_items = []
        for ip in sorted(img_dir.glob("*.png")):
            tp = tgt_dir / ip.name
            if not tp.exists():
                continue
            lp = lung_dir / ip.name if lung_dir is not None else None
            cls_items.append({"img": ip, "tgt": tp,
                              "lung": lp if lp is not None and lp.exists() else None,
                              "cls": cdir.name})
        if max_per_class and len(cls_items) > max_per_class:
            rng = np.random.default_rng(seed)
            cls_items = [cls_items[i] for i in sorted(rng.choice(len(cls_items), max_per_class, replace=False))]
        items += cls_items
    return items


def load_mask_png(path, size):
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return torch.from_numpy(np.array(m) > 127)


# ----------------------------------------------------------------------------- backbone
def build_encoder(args, device):
    if args.checkpoint:
        base.EXTRA_PREFIXES.extend(args.strip_prefix)
        sd = base.fuse_qkv(base.load_state_dict(args.checkpoint, args.ckpt_key))
        if args.base_checkpoint:
            sd = base.merge_state_dicts(
                sd, base.fuse_qkv(base.load_state_dict(args.base_checkpoint, args.base_ckpt_key)))
        model = base.TimmViTEncoder(sd, args.img_size)
        mean = torch.tensor(args.mean or [0.485, 0.456, 0.406])
        std = torch.tensor(args.std or [0.229, 0.224, 0.225])
        # many runs save "teacher_checkpoint.pth": add a short hash of the full paths so caches never collide
        import hashlib
        key = (str(Path(args.checkpoint).resolve()) + "|" + str(Path(args.base_checkpoint).resolve() if args.base_checkpoint else "")
               + "|qkv-fused")
        name = f"{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}_{hashlib.md5(key.encode()).hexdigest()[:8]}"
    else:
        model = AutoModel.from_pretrained(args.encoder)
        try:
            proc = AutoImageProcessor.from_pretrained(args.encoder)
            mean, std = torch.tensor(proc.image_mean), torch.tensor(proc.image_std)
        except Exception:
            mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
        name = Path(args.encoder).name
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    patch = getattr(model.config, "patch_size", 14)
    assert args.img_size % patch == 0, f"--img_size must be a multiple of {patch}"
    return model, mean, std, patch, name


# ----------------------------------------------------------------------------- features
def build_cache(split, items, model, enc_name, args, mean, std, patch, device):
    layer_tag = "_".join(f"L{-l}" for l in args.layers)
    tag = f"covidqu_{args.task}_{split}_{enc_name}_{args.img_size}_{layer_tag}"
    if args.max_per_class:
        tag += f"_max{args.max_per_class}"
        if args.data_seed != 0:
            tag += f"_ds{args.data_seed}"
    cache = Path(args.cache_dir) / f"{tag}.pt"
    if cache.exists():
        vprint(f"[{split}] loading cached features {cache}")
        return torch.load(cache)

    g = args.img_size // patch
    est = len(items) * len(args.layers) * 768 * g * g * 2 / 1e9
    vprint(f"[{split}] extracting {len(items)} images (~{est:.1f} GB of fp16 features)")

    feats = {l: [] for l in args.layers}
    tgts, lungs, batch = [], [], []

    def flush():
        x = torch.stack(batch).to(device)
        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
            grids = base.extract(model, x, patch, args.layers)
        for l, f in grids.items():
            feats[l].append(f.half().cpu())
        batch.clear()

    for i, it in enumerate(items):
        img = base.load_gray(it["img"])
        x = torch.from_numpy(base.resize(img, args.img_size)).float()[None].repeat(3, 1, 1)
        batch.append((x - mean[:, None, None]) / std[:, None, None])
        tgts.append(load_mask_png(it["tgt"], args.mask_size))
        lungs.append(load_mask_png(it["lung"], args.mask_size) if it["lung"] is not None
                     else torch.zeros(args.mask_size, args.mask_size, dtype=torch.bool))
        if len(batch) == args.extract_bs:
            flush()
        if (i + 1) % 500 == 0:
            vprint(f"[{split}] {i + 1}/{len(items)}")
    if batch:
        flush()

    data = {"feats": {l: torch.cat(v) for l, v in feats.items()},
            "tgt": torch.stack(tgts).to(torch.uint8),
            "lung": torch.stack(lungs).to(torch.uint8),
            "cls": [it["cls"] for it in items]}
    if args.no_cache:
        return data
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, cache)
        vprint(f"[{split}] cached -> {cache}")
    except (RuntimeError, OSError) as e:  # disk full / quota exceeded: keep going without cache
        cache.unlink(missing_ok=True)
        print(f"[{split}] WARNING could not write cache ({e.__class__.__name__}: {str(e)[:80]}). "
              f"Continuing without cache; use --cache_dir on a larger disk (e.g. scratch).")
    return data


# ----------------------------------------------------------------------------- metrics
def dice(pred, tgt, eps=1e-6):
    pred, tgt = pred.float().flatten(1), tgt.float().flatten(1)
    return (2 * (pred * tgt).sum(1) + eps) / (pred.sum(1) + tgt.sum(1) + eps)


def auroc(scores, labels):
    """Rank-based AUROC (Mann-Whitney U); no sklearn needed."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    for v in np.unique(scores):
        idx = scores == v
        ranks[idx] = ranks[idx].mean()
    return (ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg)


# ----------------------------------------------------------------------------- training
def loss_fn(logits, masks):
    ce = F.cross_entropy(logits, masks.long())
    pos = masks.flatten(1).bool().any(1)  # bool: ~pos on a uint8 tensor is NOT a negation
    if pos.any():  # soft Dice only on images that contain the target
        prob = logits.softmax(1)[:, 1][pos]
        return ce + (1 - dice(prob, masks[pos]).mean())
    return ce


@torch.no_grad()
def predict(head, feats, mask_size, bs=64):
    head.eval()
    dev = next(head.parameters()).device
    out = []
    for i in range(0, len(feats), bs):
        f = feats[i:i + bs].to(dev, non_blocking=True).float()
        logits = F.interpolate(head(f), size=mask_size, mode="bilinear", align_corners=False)
        out.append(logits.argmax(1).to(torch.uint8).cpu())
    return torch.cat(out)


def train_probe(tr_f, tr_m, va_f, va_m, seed, args, device):
    torch.manual_seed(seed)
    # keep features on the GPU if they fit comfortably (much faster than per-batch copies)
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        need = (tr_f.numel() + va_f.numel()) * tr_f.element_size()
        if need < 0.6 * free:
            tr_f, va_f, tr_m = tr_f.to(device), va_f.to(device), tr_m.to(device)
            vprint(f"  features on GPU ({need / 1e9:.1f} GB)")
    head = base.LinearHead(tr_f.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    va_pos = va_m.flatten(1).bool().any(1)
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(tr_f), device=tr_f.device)
        tot = 0.0
        for i in range(0, len(perm), args.bs):
            b = perm[i:i + args.bs]
            f = tr_f[b].to(device, non_blocking=True).float()
            m = tr_m[b].to(device, non_blocking=True)
            logits = F.interpolate(head(f), size=args.mask_size, mode="bilinear", align_corners=False)
            loss = loss_fn(logits, m)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        pred = predict(head, va_f, args.mask_size)
        va_m_cpu = va_m.cpu()
        va_pos_cpu = va_pos.cpu()
        val = (dice(pred[va_pos_cpu], va_m_cpu[va_pos_cpu]).mean().item() if va_pos_cpu.any()
               else dice(pred, va_m_cpu).mean().item())
        if val > best:
            best, best_state = val, {k: v.clone() for k, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            vprint(f"  epoch {ep:3d} | loss {tot / len(tr_f):.4f} | val Dice (positives) {100 * val:.2f}")
    head.load_state_dict(best_state)
    return head, 100 * best


def evaluate(head, te, feats, args):
    """Return {metric: (value, n)}; value in %, n = number of test images it is computed on."""
    tgt, lung = te["tgt"], te["lung"]
    pred = predict(head, feats, args.mask_size)
    pos = tgt.flatten(1).bool().any(1)  # bool, so ~pos really selects the negatives
    nan = float("nan")

    def mean_dice(sel, p=None):
        p = pred if p is None else p
        n = int(sel.sum())
        return (dice(p[sel], tgt[sel]).mean().item() * 100 if n else nan, n)

    res = {"Dice positives": mean_dice(pos)}

    if args.task == "infection":
        has_lung = lung.flatten(1).bool().any(1)
        cov = tgt.flatten(1).sum(1).float() / lung.flatten(1).sum(1).clamp(min=1).float()
        focal = pos & has_lung & (cov < args.focal_max_coverage)
        base_pos = pos & has_lung
        res[f"Dice focal (<{args.focal_max_coverage:.0%} of lung)"] = mean_dice(focal)
        res["Lung-mask baseline (positives)"] = mean_dice(base_pos, lung)
        res["Lung-mask baseline (focal)"] = mean_dice(focal, lung)
        res["median infection/lung coverage (%)"] = (
            cov[base_pos].median().item() * 100 if base_pos.any() else nan, int(base_pos.sum()))

        area = pred.flatten(1).sum(1).float()
        det = area >= args.min_area
        sens = det[pos].float().mean().item() * 100 if pos.any() else nan
        spec = (~det[~pos]).float().mean().item() * 100 if (~pos).any() else nan
        res["Detection AUROC (pred. area)"] = (100 * auroc(area.numpy(), pos.numpy()), len(pos))
        res[f"Sensitivity (area >= {args.min_area}px)"] = (sens, int(pos.sum()))
        res[f"Specificity (area >= {args.min_area}px)"] = (spec, int((~pos).sum()))
        # per class Dice (positives only exist for COVID-19)
        for c in sorted(set(te["cls"])):
            idx = torch.tensor([x == c for x in te["cls"]])
            if (idx & pos).any():
                res[f"Dice positives [{c}]"] = mean_dice(idx & pos)
    return res


def summarise(vals):
    """mean ± std across seeds, ignoring NaN (e.g. an empty focal subset)."""
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "-"
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.1f} ± {sd:.1f}"


def run_probes(data, args, device, enc_name):
    """Train one head per (layer combination, seed); return {probe name: [result per seed]}."""
    probes = []
    if args.layer_mode in ("separate", "both"):
        probes += [(l,) for l in args.layers]
    if args.layer_mode in ("concat", "both") and len(args.layers) > 1:
        probes.append(tuple(args.layers))
    nice = {-1: "last", -2: "second_last"}

    print(f"\n===== Per-seed results ({args.task}, test set) =====")
    all_res = {}
    for layers in probes:
        pname = "+".join(nice.get(l, f"layer{l}") for l in layers)
        sel = lambda s: torch.cat([data[s]["feats"][l] for l in layers], dim=1)
        tr_f, va_f, te_f = sel("train"), sel("val"), sel("test")
        vprint(f"\n----- {args.task} probe on {pname} -----")
        vprint(f"  train {len(tr_f)} | val {len(va_f)} | grid {tuple(tr_f.shape[1:])}")
        all_res[pname] = []
        for seed in args.seeds:
            vprint(f"\n  --- seed {seed} ---")
            head, val = train_probe(tr_f, data["train"]["tgt"], va_f, data["val"]["tgt"],
                                    seed, args, device)
            res = {"best val Dice (positives)": (val, len(va_f))}
            res.update(evaluate(head, data["test"], te_f, args))
            all_res[pname].append(res)
            torch.save(head.state_dict(), f"covidqu_{args.task}_head_{enc_name}_{pname}_seed{seed}.pt")
            line = f"  seed {seed:>3d}  {pname:18s} val {val:5.1f} | Dice pos {res['Dice positives'][0]:5.1f}"
            focal_key = next((k for k in res if k.startswith("Dice focal")), None)
            if focal_key:
                line += f" | focal {res[focal_key][0]:5.1f}"
                line += f" | AUROC {res['Detection AUROC (pred. area)'][0]:5.1f}"
            print(line)
            del head
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del tr_f, va_f, te_f
    return all_res


def print_summary(all_res, args, enc_name):
    print(f"\n===== COVID-QU-Ex {args.task} | encoder {enc_name} | img {args.img_size} | TEST | "
          f"{len(args.seeds)} seeds: mean ± std across seeds =====")
    seeds_txt = " ".join(str(s) for s in args.seeds)
    for pname, runs in all_res.items():
        print(f"\n[{pname}]")
        print(f"  {'metric':42s}{'mean ± std':>14s}{'n':>8s}   per seed ({seeds_txt})")
        for k in runs[0]:
            vals = [r[k][0] for r in runs]
            n = runs[0][k][1]
            per = " ".join("-" if math.isnan(v) else f"{v:.1f}" for v in vals)
            print(f"  {k:42s}{summarise(vals):>14s}{n:>8d}   {per}")


# ----------------------------------------------------------------------------- main
def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder where COVID-QU-Ex was unpacked")
    ap.add_argument("--task", choices=["infection", "lung"], default="infection")
    ap.add_argument("--classes", nargs="*", default=None,
                    help="restrict to classes, e.g. COVID-19 (default: all: COVID-19 Non-COVID Normal)")
    ap.add_argument("--max_per_class", type=int, default=None,
                    help="subsample each class per split (useful for the 34k-image lung task)")
    # backbone (same options as segmentation.py)
    ap.add_argument("--encoder", default="microsoft/rad-dino")
    ap.add_argument("--checkpoint"); ap.add_argument("--ckpt_key")
    ap.add_argument("--base_checkpoint"); ap.add_argument("--base_ckpt_key")
    ap.add_argument("--strip_prefix", nargs="*", default=[])
    ap.add_argument("--mean", type=float, nargs=3); ap.add_argument("--std", type=float, nargs=3)
    # probe
    ap.add_argument("--layers", type=int, nargs="+", default=[-1, -2])
    ap.add_argument("--layer_mode", choices=["separate", "concat", "both"], default="separate")
    ap.add_argument("--img_size", type=int, default=518,
                    help="images are 256x256; 518 matches RAD-DINO, 252 is ~native and 4x cheaper")
    ap.add_argument("--mask_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--extract_bs", type=int, default=32)
    ap.add_argument("--cache_dir", default="feature_cache",
                    help="where cached features go; point this at scratch, they are large")
    ap.add_argument("--no_cache", action="store_true", help="keep features in memory only")
    ap.add_argument("--seeds", "--seed", dest="seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="one head per seed; results reported as mean +/- std across seeds")
    ap.add_argument("--data_seed", type=int, default=0,
                    help="seed for --max_per_class subsampling (same for every run)")
    ap.add_argument("--verbose", action="store_true",
                    help="print training progress and feature-extraction logs")
    # evaluation
    ap.add_argument("--focal_max_coverage", type=float, default=0.5,
                    help="positives whose infection covers less than this fraction of the lung")
    ap.add_argument("--min_area", type=int, default=50,
                    help="predicted pixels (at --mask_size) needed to call an image 'infected'")
    args = ap.parse_args()
    VERBOSE = args.verbose

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = find_split_dirs(args.root, args.task)
    items = {s: collect_split(dirs[s], args.task, args.classes, args.max_per_class, args.data_seed)
             for s in SPLITS}
    for s in SPLITS:
        counts = {c: sum(it["cls"] == c for it in items[s]) for c in sorted({it["cls"] for it in items[s]})}
        print(f"{s:5s}: {len(items[s])} images {counts}")
        if not items[s]:
            raise SystemExit(f"No images found for split {s}")
    print(f"seeds {args.seeds}")

    model, mean, std, patch, enc_name = build_encoder(args, device)
    data = {s: build_cache(s, items[s], model, enc_name, args, mean, std, patch, device) for s in SPLITS}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    all_res = run_probes(data, args, device, enc_name)
    print_summary(all_res, args, enc_name)
    print(f"\n(heads saved as covidqu_{args.task}_head_{enc_name}_<layers>_seed<N>.pt; "
          f"use --verbose for training logs)")


if __name__ == "__main__":
    main()