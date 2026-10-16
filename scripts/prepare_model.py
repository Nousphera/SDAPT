#!/usr/bin/env python3
"""Download and validate one pinned official DINO, DINOv2, or DINOv3 ViT."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sdapt.preparation import prepare_model
from sdapt.registry import DEFAULT_MODEL, MODEL_SPECS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=MODEL_SPECS, help="Default: paper's DINOv2 ViT-B/14")
    parser.add_argument("--output", help="Override checkpoints/<model>.safetensors (paper baseline keeps dinov2-vit-b14.safetensors)")
    parser.add_argument("--check-only", action="store_true", help="Validate the existing checkpoint without downloading")
    parser.add_argument("--list-models", action="store_true", help="List supported models and artifact sizes without downloading")
    args = parser.parse_args(argv)
    if args.list_models:
        for name, spec in MODEL_SPECS.items():
            print(f"{name:30s} {spec.size / 1e9:7.3f} GB  patch={spec.patch_size:2d} blocks={spec.depth:2d} width={spec.embed_dim:4d}")
        return 0
    try:
        result = prepare_model(args.output, model=args.model, check_only=args.check_only)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Model preparation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
