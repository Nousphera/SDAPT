#!/usr/bin/env python3
"""Prepare one paper dataset, or all three if --dataset is omitted."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sdapt.data import DATASETS
from sdapt.preparation import AccessRequired, prepare_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, help="Omit to prepare all three paper domains")
    parser.add_argument("--data-root", default="data", help="Output root, default: data")
    parser.add_argument("--fmow-dir", help="Existing WILDS folder containing rgb_metadata.csv and images/")
    parser.add_argument("--quickdraw-dir", help="Existing folder containing quickdraw/ and optional official split manifests")
    parser.add_argument("--chexpert-dir", help="Authorized extracted CheXpert folder containing train.csv and train/")
    parser.add_argument("--chexlocalize-dir", help="Authorized CheXlocalize CheXpert folder containing test_labels.csv and test/")
    parser.add_argument("--check-only", action="store_true", help="Validate prepared manifests and image paths without downloading")
    args = parser.parse_args(argv)
    failures = []
    for name in (args.dataset,) if args.dataset else DATASETS:
        print(f"Preparing {name}...", flush=True)
        try:
            result = prepare_dataset(name, args.data_root, fmow_dir=args.fmow_dir, quickdraw_dir=args.quickdraw_dir,
                                     chexpert_dir=args.chexpert_dir, chexlocalize_dir=args.chexlocalize_dir, check_only=args.check_only)
            print(f"{name}: ready ({result['counts']['train']:,} train, {result['counts']['test']:,} test)", flush=True)
        except (AccessRequired, OSError, ValueError, RuntimeError) as error:
            failures.append(name)
            print(f"{name}: {error}", file=sys.stderr, flush=True)
    if failures:
        print("Preparation incomplete for: " + ", ".join(failures), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
