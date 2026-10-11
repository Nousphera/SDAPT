#!/usr/bin/env python
"""
Linear-probe segmentation of clavicles (or heart / lungs) on JSRT + SCR with a frozen ViT.

Put this file next to seg_cov.py (the COVID-QU-Ex probe) and segmentation.py: it reuses the
backbone loading (--checkpoint / --base_checkpoint / HF --encoder), the probe training and the
metrics from there, so results are directly comparable with your COVID-QU-Ex runs.

Data
  --jsrt    folder with the 247 JSRT images: JPCLN001..154 + JPCNN001..093 (.png/.jpg or raw .IMG)
  --points  folder with the unzipped SCR points.zip: one file per image (JPCLN001.pfs ...) with the
            hand-drawn outline of each structure ([Label=right lung], [Label=left clavicle], ...,
            coordinates on the 1024x1024 grid). Each outline is filled to get the mask, so structures
            are named explicitly and overlaps (clavicle over lung) are kept.
  Left/right follow SCR / radiology: "left" = patient's left = right side of the image.

Protocol (official SCR): fold 1 = odd-numbered images (124), fold 2 = even-numbered (123).
Train on one fold, test on the other, both ways, so every image is tested once.
20% of the training fold is held out to pick the best epoch. Reported:
  * Dice and Jaccard over all 247 test predictions, per fold, and per side (left/right)
  * per-image scores in a CSV (for a paired comparison of two backbones)
Overlay images (lungs blue, heart green, clavicles red) are written to --check_dir: look at them once.

Multiple seeds
  --seeds 0 1 2      the whole 2-fold protocol is run once per seed (default 0 1 2); results are
                     reported as mean +/- std ACROSS SEEDS. Features are extracted once. The SCR
                     folds are fixed; a seed changes the 20% validation hold-out, the head
                     initialisation and the batch order.
  --val_seed 0       keep the SAME validation hold-out for every seed (only head init / order vary)
  --verbose          show training progress (per-epoch loss / val Dice) and extraction logs

Example
  python src/scr_seg_probe.py --jsrt /scratch-shared/svhonk/images/images --points /scratch-shared/svhonk/points \
      --structure clavicle --checkpoint lora.pth --base_checkpoint dino.pth --layer_mode both --seeds 0 1 2
"""
import argparse
import contextlib
import csv
import inspect
import io
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

try:  # the COVID-QU-Ex probe script, whatever you named it
    import utils.seg_evaluation.seg_cov as cq  # build_encoder, train_probe, predict, dice
except ImportError:
    import covid_seg_probe as cq

base = cq.base

PARTS = {"clavicle": ("left clavicle", "right clavicle"),
         "heart": ("heart",),
         "lung": ("left lung", "right lung")}
ALL_PARTS = ("right lung", "left lung", "heart", "right clavicle", "left clavicle")
IMG_EXT = {".img", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

VERBOSE = False


def vprint(*a, **k):
    """Print only with --verbose."""
    if VERBOSE:
        print(*a, **k)


# ----------------------------------------------------------------------------- reading files
def index_files(root, ext=None):
    """{case id (e.g. JPCLN001): path} for JSRT-named files under root."""
    idx = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.stem.upper().startswith("JPC") and (ext is None or p.suffix.lower() in ext):
            s = p.stem.upper()
            if s not in idx or p.suffix.lower() == ".img":  # prefer original raw images
                idx[s] = p
    return idx


def read_image(path):
    if path.suffix.lower() == ".img":  # JSRT raw: square, 16-bit big-endian, 12 bits used
        raw = np.fromfile(path, dtype=">u2")
        side = int(round(np.sqrt(raw.size)))
        if side * side != raw.size:
            raise SystemExit(f"{path}: unexpected raw size {raw.size}")
        return raw.reshape(side, side).astype(np.float32)
    a = np.array(Image.open(path)).astype(np.float32)
    return a[..., :3].mean(-1) if a.ndim == 3 else a


def read_points(path):
    """SCR point file -> {structure: (N, 2) array of (x, y)}; the '... fixed' helper sets are skipped."""
    out, label, pts = {}, None, []
    for line in Path(path).read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("[Label="):
            label, pts = line[len("[Label="):].rstrip("]").strip().lower(), []
        elif label is not None and line.startswith("{") and "," in line:
            x, y = line.strip("{},").split(",")[:2]
            pts.append((float(x), float(y)))
        elif label is not None and line.startswith("}"):
            if not label.endswith("fixed") and len(pts) >= 3:
                out[label] = np.array(pts)
            label = None
    return out


def outline_masks(path):
    """Fill every outline -> {structure: bool mask}; grid 1024 (2048 if coordinates go beyond)."""
    shapes = read_points(path)
    size = 2048 if max(p.max() for p in shapes.values()) > 1030 else 1024
    masks = {}
    for name, pts in shapes.items():
        im = Image.new("L", (size, size), 0)
        ImageDraw.Draw(im).polygon([(float(x), float(y)) for x, y in pts], fill=1, outline=1)
        masks[name] = np.array(im, bool)
    return masks


def resize_mask(m, size):
    return torch.from_numpy(np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 127)


# ----------------------------------------------------------------------------- dataset
def make_png(it, lung, args):
    """8-bit 1024x1024 PNG (same grid as the outlines); lungs (air) must be dark, else invert."""
    a = read_image(it["img"])
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    a = np.asarray(Image.fromarray(a).resize((1024, 1024), Image.BOX))
    inv = args.invert == "yes"
    if args.invert == "auto" and lung is not None:
        lung = np.array(Image.fromarray(lung.astype(np.uint8)).resize((1024, 1024), Image.NEAREST)) > 0
        if lung.any() and (~lung).any():
            inv = a[lung].mean() > a[~lung].mean()
    if inv:
        a = 1 - a
    Image.fromarray((a * 255).round().astype(np.uint8)).save(it["png"])
    return inv


def save_check(it, masks, args):
    """Gray image with the structures drawn on top: lungs blue, heart green, clavicles red."""
    g = np.array(Image.open(it["png"]).convert("L").resize((1024, 1024))).astype(np.float32)
    rgb = np.stack([g] * 3, -1)
    colors = {"lung": (40, 120, 255), "heart": (40, 220, 80), "clavicle": (255, 40, 40)}
    for part, m in masks.items():
        m = np.array(Image.fromarray(m.astype(np.uint8)).resize((1024, 1024), Image.NEAREST)) > 0
        rgb[m] = 0.55 * rgb[m] + 0.45 * np.array(colors[part.split()[-1]], np.float32)
    out = Path(args.check_dir) / f"check_{it['stem']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(out)
    return out


def load_dataset(items, need, args):
    """Build target masks (+ per-side masks), convert images to PNG once, write check images."""
    Path(args.png_dir).mkdir(parents=True, exist_ok=True)
    tgt, sides, inverted, converted, checks = [], {p: [] for p in need if len(need) > 1}, 0, 0, []
    for i, it in enumerate(items):
        masks = {p: m for p, m in outline_masks(it["points"]).items() if p in ALL_PARTS}
        it["png"] = Path(args.png_dir) / f"{it['stem']}.png"
        if args.rebuild_png or not it["png"].exists():
            lung = masks["right lung"] | masks["left lung"] if {"right lung", "left lung"} <= set(masks) else None
            inverted += make_png(it, lung, args)
            converted += 1
        t = np.zeros(next(iter(masks.values())).shape, bool)
        for p in need:
            t |= masks[p]
        tgt.append(resize_mask(t, args.mask_size))
        for p in sides:
            sides[p].append(resize_mask(masks[p], args.mask_size))
        if i < args.n_check:
            checks.append(save_check(it, masks, args))
    if converted:
        print(f"converted {converted} images -> {args.png_dir} ({inverted} inverted to dark-lung display)")
        if 0 < inverted < converted:
            print("  WARNING: only some images were inverted; check a few PNGs by eye")
    if checks:
        vprint(f"check images (lungs blue, heart green, clavicles red): {checks[0].parent}/check_*.png")
    return torch.stack(tgt).to(torch.uint8), {p: torch.stack(v) for p, v in sides.items()}


# ----------------------------------------------------------------------------- features
def extract_all(items, model, enc_name, args, mean, std, patch, device):
    stems = [it["stem"] for it in items]
    layer_tag = "_".join(f"L{-l}" for l in args.layers)
    cache = Path(args.cache_dir) / f"jsrt_{enc_name}_{args.img_size}_{layer_tag}.pt"
    if cache.exists() and not args.no_cache:
        d = torch.load(cache)
        if d["stems"] == stems:
            vprint(f"loading cached features {cache}")
            return d["feats"]
    vprint(f"extracting {len(items)} images")
    feats, batch = {l: [] for l in args.layers}, []

    def flush():
        x = torch.stack(batch).to(device)
        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
            grids = base.extract(model, x, patch, args.layers)
        for l, f in grids.items():
            feats[l].append(f.half().cpu())
        batch.clear()

    for it in items:
        img = base.load_gray(it["png"])
        x = torch.from_numpy(base.resize(img, args.img_size)).float()[None].repeat(3, 1, 1)
        batch.append((x - mean[:, None, None]) / std[:, None, None])
        if len(batch) == args.extract_bs:
            flush()
    if batch:
        flush()
    feats = {l: torch.cat(v) for l, v in feats.items()}
    if not args.no_cache:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"stems": stems, "feats": feats}, cache)
            vprint(f"cached -> {cache}")
        except (RuntimeError, OSError) as e:
            cache.unlink(missing_ok=True)
            print(f"WARNING could not write cache ({e.__class__.__name__}); continuing without it")
    return feats


# ----------------------------------------------------------------------------- training
_TRAIN_TAKES_SEED = "seed" in inspect.signature(cq.train_probe).parameters


def train(tr_f, tr_m, va_f, va_m, seed, args, device):
    """Call the COVID script's train_probe (new signature with seed, or old one using args.seed);
    without --verbose its per-epoch output is swallowed."""
    def call():
        if _TRAIN_TAKES_SEED:
            return cq.train_probe(tr_f, tr_m, va_f, va_m, seed, args, device)
        args.seed = seed
        return cq.train_probe(tr_f, tr_m, va_f, va_m, args, device)
    if VERBOSE:
        return call()
    with contextlib.redirect_stdout(io.StringIO()):
        return call()


# ----------------------------------------------------------------------------- metrics
def side_dice(pred, part_mask):
    """Per-image Dice for one side (e.g. left clavicle): prediction restricted to the image half
    that contains this part, against this part's mask. NaN where the part is absent."""
    W, out = pred.shape[-1], []
    for p, m in zip(pred, part_mask):
        xs = m.nonzero()[:, 1]
        if len(xs) == 0:
            out.append(float("nan"))
            continue
        keep = torch.zeros_like(p)
        if xs.float().mean() < W / 2:
            keep[:, : W // 2] = 1
        else:
            keep[:, W // 2:] = 1
        out.append(cq.dice((p * keep)[None], m[None]).item())
    return torch.tensor(out)


def jaccard_from_dice(d):  # both in %
    return 100 * d / (200 - d)


def nanmean(t):
    t = t[~t.isnan()]
    return (t.mean().item() if len(t) else float("nan")), len(t)


def summarise(vals):
    """mean ± std across seeds, ignoring NaN."""
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return "-"
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.2f} ± {sd:.2f}"


def run_seed(X, tgt, part_masks, folds, seed, args, device, enc_name, pname, out_dir):
    """Full 2-fold SCR protocol for one seed -> ({metric: (value, n)}, {column: per-image tensor})."""
    preds, vals, n_vals = torch.zeros_like(tgt), [], []
    split_seed = seed if args.val_seed is None else args.val_seed
    for train_fold, test_fold in ((1, 2), (2, 1)):
        tr_all = (folds == train_fold).nonzero().flatten()
        g = torch.Generator().manual_seed(split_seed)
        perm = tr_all[torch.randperm(len(tr_all), generator=g)]
        n_va = max(1, int(round(args.val_frac * len(perm))))
        va, tr = perm[:n_va], perm[n_va:]
        te = (folds == test_fold).nonzero().flatten()
        vprint(f"\n----- seed {seed} | {args.structure} probe on {pname} | train fold {train_fold} "
               f"({len(tr)} train, {len(va)} val) -> test fold {test_fold} ({len(te)}) -----")
        head, val = train(X[tr], tgt[tr], X[va], tgt[va], seed, args, device)
        vals.append(val)
        n_vals.append(len(va))
        preds[te] = cq.predict(head, X[te], args.mask_size)
        torch.save(head.state_dict(),
                   out_dir / f"scr_{args.structure}_head_{enc_name}_{pname}_fold{train_fold}_seed{seed}.pt")
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    d = cq.dice(preds, tgt) * 100
    side = {p: side_dice(preds, pm) * 100 for p, pm in part_masks.items()}
    res = {"best val Dice, fold-1 model": (vals[0], n_vals[0]),
           "best val Dice, fold-2 model": (vals[1], n_vals[1]),
           "Dice (all images)": (d.mean().item(), len(d)),
           "Jaccard (all images)": (jaccard_from_dice(d).mean().item(), len(d))}
    for f in (1, 2):
        res[f"Dice test fold {f}"] = (d[folds == f].mean().item(), int((folds == f).sum()))
    for p, v in side.items():
        res[f"Dice {p}"] = nanmean(v)
    per_image = {"dice": d}
    per_image.update({f"dice_{p.replace(' ', '_')}": v for p, v in side.items()})
    return res, per_image


def write_csv(path, items, per_image_runs, seeds):
    """One row per image: every per-image score for each seed plus its mean over seeds."""
    cols = list(per_image_runs[0])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["image", "fold"]
        for c in cols:
            header += [f"{c}_seed{s}" for s in seeds] + [f"{c}_mean"]
        w.writerow(header)
        for i, it in enumerate(items):
            row = [it["stem"], it["fold"]]
            for c in cols:
                v = [run[c][i].item() for run in per_image_runs]
                good = [x for x in v if not math.isnan(x)]
                row += [f"{x:.3f}" for x in v] + [f"{np.mean(good):.3f}" if good else "nan"]
            w.writerow(row)


# ----------------------------------------------------------------------------- main
def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsrt", required=True, help="folder with the JSRT images (.png/.jpg or raw .IMG)")
    ap.add_argument("--points", required=True, help="folder with the unzipped SCR points.zip")
    ap.add_argument("--structure", choices=list(PARTS), default="clavicle")
    ap.add_argument("--png_dir", default="jsrt_png", help="where converted 8-bit PNGs are written")
    ap.add_argument("--check_dir", default="scr_check", help="where the overlay check images go")
    ap.add_argument("--n_check", type=int, default=5, help="number of check images")
    ap.add_argument("--rebuild_png", action="store_true")
    ap.add_argument("--invert", choices=["auto", "yes", "no"], default="auto",
                    help="auto: invert images whose lungs are brighter than the rest")
    ap.add_argument("--val_frac", type=float, default=0.2, help="part of the training fold used for epoch selection")
    ap.add_argument("--out_dir", default=".", help="where heads and per-image CSVs are saved")
    # backbone (same options as covid_seg_probe.py / segmentation.py)
    ap.add_argument("--encoder", default="microsoft/rad-dino")
    ap.add_argument("--checkpoint"); ap.add_argument("--ckpt_key")
    ap.add_argument("--base_checkpoint"); ap.add_argument("--base_ckpt_key")
    ap.add_argument("--strip_prefix", nargs="*", default=[])
    ap.add_argument("--mean", type=float, nargs=3); ap.add_argument("--std", type=float, nargs=3)
    # probe
    ap.add_argument("--layers", type=int, nargs="+", default=[-1, -2])
    ap.add_argument("--layer_mode", choices=["separate", "concat", "both"], default="separate")
    ap.add_argument("--img_size", type=int, default=518)
    ap.add_argument("--mask_size", type=int, default=512,
                    help="clavicles are thin; 512 keeps their edges (256 to match the COVID runs)")
    ap.add_argument("--epochs", type=int, default=150, help="small dataset: ~6 steps per epoch")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--extract_bs", type=int, default=16)
    ap.add_argument("--cache_dir", default="feature_cache")
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--seeds", "--seed", dest="seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="one full 2-fold run per seed; results reported as mean +/- std across seeds")
    ap.add_argument("--val_seed", type=int, default=None,
                    help="use this seed for the validation hold-out in every run (default: each "
                         "run uses its own seed for the hold-out too)")
    ap.add_argument("--verbose", action="store_true",
                    help="print training progress and feature-extraction logs")
    args = ap.parse_args()
    VERBOSE = args.verbose
    if hasattr(cq, "VERBOSE"):
        cq.VERBOSE = args.verbose

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, points = index_files(args.jsrt, IMG_EXT), index_files(args.points)
    stems = sorted(set(images) & set(points))
    print(f"JSRT images: {len(images)} | point files: {len(points)} | matched: {len(stems)}")
    if not stems:
        raise SystemExit("No matching image/point pairs. File names must match, e.g. JPCLN001.png / JPCLN001.pfs")
    need = PARTS[args.structure]
    names = read_points(points[stems[0]])
    vprint(f"structures in {points[stems[0]].name}: {sorted(names)}")
    missing = [p for p in need if p not in names]
    if missing:
        raise SystemExit(f"{missing} not found in the point files (labels found: {sorted(names)})")

    # official SCR folds: odd image numbers -> fold 1, even -> fold 2
    items = [{"stem": s, "img": images[s], "points": points[s], "fold": 1 if int(s[-3:]) % 2 else 2} for s in stems]
    folds = torch.tensor([it["fold"] for it in items])
    val_desc = (f"fixed val hold-out (seed {args.val_seed})" if args.val_seed is not None
                else "val hold-out re-drawn per seed")
    print(f"fold 1 (odd): {(folds == 1).sum().item()} | fold 2 (even): {(folds == 2).sum().item()} "
          f"| seeds {args.seeds} | {val_desc}")

    tgt, part_masks = load_dataset(items, need, args)
    print(f"mean {args.structure} area: {100 * tgt.float().mean():.2f}% of the image "
          f"| images with empty {args.structure} mask: {(~tgt.flatten(1).bool().any(1)).sum().item()}")

    model, mean, std, patch, enc_name = cq.build_encoder(args, device)
    feats = extract_all(items, model, enc_name, args, mean, std, patch, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    probes = []
    if args.layer_mode in ("separate", "both"):
        probes += [(l,) for l in args.layers]
    if args.layer_mode in ("concat", "both") and len(args.layers) > 1:
        probes.append(tuple(args.layers))
    nice = {-1: "last", -2: "second_last"}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== Per-seed results ({args.structure}, all {len(items)} test predictions) =====")
    all_res, csvs = {}, {}
    for layers in probes:
        pname = "+".join(nice.get(l, f"layer{l}") for l in layers)
        X = torch.cat([feats[l] for l in layers], dim=1)
        runs, per_image_runs = [], []
        for seed in args.seeds:
            res, per_image = run_seed(X, tgt, part_masks, folds, seed, args, device, enc_name, pname, out_dir)
            runs.append(res)
            per_image_runs.append(per_image)
            print(f"  seed {seed:>3d}  {pname:18s} Dice {res['Dice (all images)'][0]:6.2f} | "
                  f"Jaccard {res['Jaccard (all images)'][0]:6.2f} | "
                  f"fold 1 {res['Dice test fold 1'][0]:6.2f} | fold 2 {res['Dice test fold 2'][0]:6.2f}")
        csv_path = out_dir / f"scr_{args.structure}_{enc_name}_{pname}_per_image.csv"
        write_csv(csv_path, items, per_image_runs, args.seeds)
        all_res[pname], csvs[pname] = runs, csv_path
        del X

    print(f"\n===== JSRT/SCR {args.structure} | encoder {enc_name} | img {args.img_size} "
          f"| mask {args.mask_size} | 2-fold CV (official SCR folds) "
          f"| {len(args.seeds)} seeds: mean ± std across seeds =====")
    seeds_txt = " ".join(str(s) for s in args.seeds)
    for pname, runs in all_res.items():
        print(f"\n[{pname}]")
        print(f"  {'metric':34s}{'mean ± std':>16s}{'n':>6s}   per seed ({seeds_txt})")
        for k in runs[0]:
            vals = [r[k][0] for r in runs]
            per = " ".join("-" if math.isnan(v) else f"{v:.2f}" for v in vals)
            print(f"  {k:34s}{summarise(vals):>16s}{runs[0][k][1]:>6d}   {per}")
        print(f"  per-image scores: {csvs[pname]}")
    print(f"\n(use --verbose for training logs)")


if __name__ == "__main__":
    main()