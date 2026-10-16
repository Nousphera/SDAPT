"""Download official public artifacts and prepare auditable SDAPT manifests.

No dataset is substituted when its source is unavailable. CheXpert and
CheXlocalize require access obtained from their publishers before local import.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .data import CHEXPERT_LABELS, DATASETS, FMOW_CLASSES, image_path, load_dataset, validate_target
from .registry import DEFAULT_MODEL, get_model_spec

FMOW_URL = "https://worksheets.codalab.org/rest/bundles/0xaec91eb7c9d548ebb15e1b5e60f966ab/contents/blob/"
QUICKDRAW_URL = "https://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip"
QUICKDRAW_MANIFEST_URL = "https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_{split}.txt"
QUICKDRAW_MANIFEST_SHA256 = {
    "train": "3a5edd3bc215772010eebc87f158707703dd7e48091d13b816af81326a9d1c0f",
    "test": "2a00a60650a453c65c356da748fec54084a197272137c440e7d52d08947357bf",
}
# Backward-compatible names for the paper baseline; all variants use registry specs.
MODEL_ID = get_model_spec().hf_repo
MODEL_REVISION = get_model_spec().revision
MODEL_SHA256 = get_model_spec().sha256
MODEL_SIZE = get_model_spec().size
MODEL_URL = get_model_spec().url
USER_AGENT = "SDAPT-research-preparation/1.0"
CHEXPERT_ACCESS = """CheXpert requires publisher-authorized access; no images were downloaded.
1. Obtain CheXpert training data from https://stanfordmlgroup.github.io/competitions/chexpert/
   and place its extracted folder (train.csv and train/) at data/chexpert/raw/CheXpert-v1.0-small/.
2. Obtain CheXlocalize from https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c
   following https://github.com/rajpurkarlab/cheXlocalize/blob/master/download_instructions.md .
   Place its CheXpert folder (test_labels.csv and test/) at data/chexpert/raw/CheXlocalize/CheXpert/.
3. Rerun: python scripts/prepare_data.py --dataset chexpert
Existing downloads elsewhere can be used without copying:
  python scripts/prepare_data.py --dataset chexpert --chexpert-dir /path/to/CheXpert-v1.0-small --chexlocalize-dir /path/to/CheXlocalize/CheXpert
Only CheXpert train and CheXlocalize test are used; validation images are excluded.
Unknown/uncertain labels are retained as -1 and must be masked in supervised losses/metrics."""


class AccessRequired(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def download(url, destination, *, sha256=None, expected_size=None, attempts=3):
    """Resume HTTP transfers; publish only complete, optionally hash-verified files.

    Range responses are checked, validators prevent splicing changed resources,
    and a server that ignores Range causes a fresh transfer. Partial files survive
    interruption. Checksummed artifacts can safely resume without an HTTP ETag.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_name(destination.name + ".download.json")
    part = destination.with_name(destination.name + ".part")
    partial_receipt = part.with_name(part.name + ".json")
    if destination.exists():
        if expected_size is not None and destination.stat().st_size != expected_size:
            raise ValueError(f"Existing file has wrong size: {destination}; remove it to download again")
        if sha256 is not None:
            if sha256_file(destination) != sha256:
                raise ValueError(f"SHA256 mismatch for existing file {destination}")
            return destination
        if receipt.exists():
            recorded = json.loads(receipt.read_text())
            if recorded.get("url") == url and recorded.get("sha256") == sha256_file(destination):
                return destination
        raise ValueError(f"Existing file has no matching download receipt: {destination}; use a source directory or move the file")
    for attempt in range(attempts):
        try:
            previous = json.loads(partial_receipt.read_text()) if partial_receipt.exists() else {}
            offset = part.stat().st_size if part.exists() else 0
            if previous.get("url", url) != url or (offset and not sha256 and not previous.get("validator")):
                offset = 0
            if expected_size is not None:
                if offset > expected_size:
                    offset = 0
                remaining = expected_size - offset
                if shutil.disk_usage(destination.parent).free < remaining + 64 * 1024**2:
                    raise OSError(f"Insufficient free disk space for {destination}: need {remaining/1024**3:.1f} GiB remaining plus 64 MiB reserve")
            headers = {"User-Agent": USER_AGENT}
            if offset:
                headers["Range"] = f"bytes={offset}-"
                if previous.get("validator"):
                    headers["If-Range"] = previous["validator"]
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                status_code = getattr(response, "status", None)
                if status_code == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset:
                        raise ValueError("Invalid Content-Range received during resumed download")
                    total = int(match[3])
                else:
                    offset = 0
                    total = int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else expected_size
                if expected_size is not None and total is not None and total != expected_size:
                    raise ValueError(f"Unexpected download size for {url}: {total}, expected {expected_size}")
                if total is not None and shutil.disk_usage(destination.parent).free < total - offset + 64 * 1024**2:
                    raise OSError(f"Insufficient free disk space for {destination}: need at least {(total-offset)/1024**3:.1f} GiB plus extraction space")
                validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
                _atomic_json(partial_receipt, {"url": url, "validator": validator, "size": total})
                print(f"Downloading {destination.name}" + (f" (resume at {offset:,} bytes)" if offset else ""), flush=True)
                transferred, last_report = offset, time.monotonic()
                with part.open("ab" if offset else "wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        transferred += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 10:
                            progress = f"{transferred/total:.1%}" if total else f"{transferred/1024**3:.2f} GiB"
                            print(f"  {destination.name}: {progress} ({transferred:,} bytes)", flush=True)
                            last_report = now
                if total is not None and part.stat().st_size != total:
                    raise OSError(f"Incomplete download: {part.stat().st_size} of {total} bytes")
            actual_hash = sha256_file(part)
            if sha256 is not None and actual_hash != sha256:
                raise ValueError(f"SHA256 mismatch for {destination}; partial file retained for inspection")
            part.replace(destination)
            _atomic_json(receipt, {"url": url, "bytes": destination.stat().st_size, "sha256": actual_hash})
            partial_receipt.unlink(missing_ok=True)
            return destination
        except urllib.error.HTTPError as error:
            # A completed partial transfer can receive 416 on the next attempt.
            if error.code == 416 and part.exists() and sha256 and sha256_file(part) == sha256:
                part.replace(destination)
                _atomic_json(receipt, {"url": url, "bytes": destination.stat().st_size, "sha256": sha256})
                partial_receipt.unlink(missing_ok=True)
                return destination
            if error.code < 500 or attempt == attempts - 1:
                raise
        except (OSError, urllib.error.URLError) as error:
            if "Insufficient free disk" in str(error) or attempt == attempts - 1:
                raise
        time.sleep(attempt + 1)
    raise RuntimeError(f"Download failed: {url}")


def _archive_path(destination, name):
    name = name.rstrip("/")
    if not name or name == ".":
        return Path(destination).resolve()
    return image_path(Path(destination), name)


def safe_extract(archive, destination):
    """Validate every entry before writing; reject links, traversal and devices.

    Regular files are published atomically. Existing files of the expected size
    are retained, so extraction can be resumed after an interrupted preparation.
    """
    archive, destination = Path(archive), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    def copy(source, target, size):
        if target.is_file() and target.stat().st_size == size:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".extracting")
        with temporary.open("wb") as output:
            shutil.copyfileobj(source, output, 1024 * 1024)
        if temporary.stat().st_size != size:
            raise OSError(f"Incomplete extracted file: {target}")
        temporary.replace(target)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            entries = source.infolist()
            for entry in entries:
                _archive_path(destination, entry.filename)
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                    raise ValueError(f"Unsafe archive link or special file: {entry.filename}")
            required = sum(e.file_size for e in entries if not e.is_dir() and not _archive_path(destination, e.filename).is_file())
            if shutil.disk_usage(destination).free < required + 64 * 1024**2:
                raise OSError(f"Insufficient free disk space to extract {archive}: need {required/1024**3:.1f} GiB")
            for entry in entries:
                if not entry.is_dir():
                    with source.open(entry) as stream:
                        copy(stream, _archive_path(destination, entry.filename), entry.file_size)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as source:
            entries = source.getmembers()
            for entry in entries:
                _archive_path(destination, entry.name)
                if not (entry.isfile() or entry.isdir()):
                    raise ValueError(f"Unsafe archive link or special file: {entry.name}")
            required = sum(e.size for e in entries if e.isfile() and not _archive_path(destination, e.name).is_file())
            if shutil.disk_usage(destination).free < required + 64 * 1024**2:
                raise OSError(f"Insufficient free disk space to extract {archive}: need {required/1024**3:.1f} GiB")
            for entry in entries:
                if entry.isfile():
                    with source.extractfile(entry) as stream:
                        copy(stream, _archive_path(destination, entry.name), entry.size)
    else:
        raise ValueError(f"Not a supported tar/zip archive: {archive}")


def fmow_records(metadata_csv):
    """Reproduce WILDS v1.1 split_scheme='official' (time_after_2016)."""
    records = {"train": [], "test": []}
    classes = {name: i for i, name in enumerate(FMOW_CLASSES)}
    with Path(metadata_csv).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"category", "split", "timestamp"}.issubset(reader.fieldnames or []):
            raise ValueError("fMoW metadata is missing category/split/timestamp columns")
        for index, row in enumerate(reader):
            timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            timestamp = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp.astimezone(timezone.utc)
            split = None
            if row["split"] == "train" and timestamp < datetime(2013, 1, 1, tzinfo=timezone.utc):
                split = "train"
            elif row["split"] == "test" and timestamp >= datetime(2016, 1, 1, tzinfo=timezone.utc):
                split = "test"
            if split:
                records[split].append({"path": f"images/rgb_img_{index}.png", "target": classes[row["category"]]})
    return records


def quickdraw_records(manifest):
    records, classes, seen = [], {}, set()
    with Path(manifest).open() as stream:
        for number, line in enumerate(stream, 1):
            path, label = line.strip().rsplit(maxsplit=1)
            label = int(label)
            parts = PurePosixPath(path).parts
            if len(parts) != 3 or parts[0] != "quickdraw" or label < 0:
                raise ValueError(f"Invalid DomainNet manifest row {number}: {line.strip()}")
            if path in seen:
                raise ValueError(f"duplicate QuickDraw image: {path}")
            if label in classes and classes[label] != parts[1]:
                raise ValueError(f"Inconsistent QuickDraw label {label}")
            seen.add(path)
            classes[label] = parts[1]
            records.append({"path": path, "target": label})
    return records, classes


def chexpert_records(labels_csv, source_dir, split):
    """Import standard CheXpert numeric labels; uncertain AND blank values are masked."""
    if split not in ("train", "test"):
        raise ValueError("CheXpert split must be train or test")
    source_dir = Path(source_dir).resolve()
    records = []
    with Path(labels_csv).open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if not set(CHEXPERT_LABELS).issubset(fields):
            raise ValueError(f"Missing required CheXpert columns: {sorted(set(CHEXPERT_LABELS)-fields)}")
        path_column = "Path" if "Path" in fields else "path" if "path" in fields else None
        if path_column is None:
            raise ValueError("CheXpert CSV requires a Path column")
        for number, row in enumerate(reader, 2):
            parts = PurePosixPath(row[path_column]).parts
            if ".." in parts or PurePosixPath(row[path_column]).is_absolute():
                raise ValueError(f"Unsafe CheXpert path at row {number}")
            # Official train CSV prefixes its package directory; test CSV may prefix CheXpert.
            if split not in parts:
                raise ValueError(f"Expected {split}/ image path in {labels_csv}:{number}")
            relative = str(PurePosixPath(*parts[parts.index(split):]))
            path = image_path(source_dir, relative)
            if not path.is_file():
                raise FileNotFoundError(f"Missing CheXpert image: {path}")
            target = []
            for name in CHEXPERT_LABELS:
                value = row[name].strip()
                numeric = -1.0 if value.lower() in ("", "nan", "none") else float(value)
                if numeric not in (-1, 0, 1):
                    raise ValueError(f"Invalid {name} label at {labels_csv}:{number}: {value}")
                target.append(int(numeric))
            patient = next((p for p in parts if re.fullmatch(r"patient\d+", p)), None)
            if patient is None:
                raise ValueError(f"No patient identifier in CheXpert path: {row[path_column]}")
            records.append({"path": relative, "target": target, "patient": patient})
    return records


def write_manifests(folder, splits, classes, multilabel, *, image_roots=None, provenance=None):
    """Validate both complete splits, then publish manifests and their receipt."""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    roots = {split: Path((image_roots or {}).get(split, folder)).resolve() for split in ("train", "test")}
    paths, patients = {}, {}
    for split in ("train", "test"):
        if not splits[split]:
            raise ValueError(f"Empty {split} split")
        paths[split], patients[split] = set(), set()
        for row in splits[split]:
            path = image_path(roots[split], row["path"])
            validate_target(row["target"], classes, multilabel)
            if path in paths[split]:
                raise ValueError(f"duplicate image in {split}: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"Missing {split} image: {path}")
            paths[split].add(path)
            if row.get("patient"):
                patients[split].add(row["patient"])
    if paths["train"] & paths["test"]:
        raise ValueError("Train/test image overlap detected")
    if patients["train"] & patients["test"]:
        raise ValueError("Train/test patient overlap detected")
    hashes = {}
    for split in ("train", "test"):
        target = folder / f"{split}.jsonl"
        temporary = target.with_suffix(".jsonl.tmp")
        with temporary.open("w") as stream:
            for row in splits[split]:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        hashes[split] = sha256_file(temporary)
        temporary.replace(target)
    metadata = {
        "schema_version": 1, "classes": list(classes), "multilabel": multilabel,
        "image_roots": {split: os.path.relpath(root, folder) if root.is_relative_to(folder) else str(root) for split, root in roots.items()},
        "counts": {split: len(rows) for split, rows in splits.items()},
        "manifest_sha256": hashes, "sources": provenance or {},
        "unknown_label": -1 if multilabel else None,
    }
    _atomic_json(folder / "metadata.json", metadata)
    return metadata


def _find_root(base, marker, candidates):
    base = Path(base)
    for candidate in (base, *(base / name for name in candidates)):
        if (candidate / marker).is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find {marker} under {base}")


def prepare_dataset(name, root="data", *, fmow_dir=None, quickdraw_dir=None, chexpert_dir=None, chexlocalize_dir=None, check_only=False):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    folder = Path(root).expanduser().resolve() / name
    explicit = any(value is not None for value in (fmow_dir, quickdraw_dir, chexpert_dir, chexlocalize_dir))
    if check_only or ((folder / "metadata.json").is_file() and not explicit):
        datasets = {split: load_dataset(name, root, split) for split in ("train", "test")}
        if set(datasets["train"].paths) & set(datasets["test"].paths):
            raise ValueError("Train/test image overlap detected")
        patient_sets = [{r["patient"] for r in datasets[s].records if r.get("patient")} for s in ("train", "test")]
        if patient_sets[0] & patient_sets[1]:
            raise ValueError("Train/test patient overlap detected")
        return json.loads((folder / "metadata.json").read_text())
    if name == "fmow":
        raw = Path(fmow_dir).expanduser() if fmow_dir else folder / "raw"
        archive = folder / "downloads/fmow_v1.1.tar.gz"
        if not fmow_dir and archive.is_file():
            # Metadata may have been extracted before an interrupted image transfer.
            safe_extract(archive, raw)
        try:
            source = _find_root(raw, "rgb_metadata.csv", ("fmow_v1.1", "fmow"))
        except FileNotFoundError:
            if fmow_dir:
                raise
            try:
                # WILDS compressed_size is a progress estimate, not a verified byte count.
                archive = download(FMOW_URL, archive)
            except (OSError, urllib.error.URLError) as error:
                raise RuntimeError(f"Official WILDS fMoW download failed: {error}. Reuse a complete WILDS v1.1 download with --fmow-dir /path/to/fmow_v1.1 (rgb_metadata.csv and images/). Source: {FMOW_URL}") from error
            safe_extract(archive, raw)
            source = _find_root(raw, "rgb_metadata.csv", ("fmow_v1.1", "fmow"))
        return write_manifests(folder, fmow_records(source / "rgb_metadata.csv"), FMOW_CLASSES, False,
                               image_roots={"train": source, "test": source},
                               provenance={"dataset": "WILDS fMoW 1.1", "split_scheme": "official (train before 2013; test from 2016)", "url": FMOW_URL, "metadata_sha256": sha256_file(source / "rgb_metadata.csv")})
    if name == "quickdraw":
        source = Path(quickdraw_dir).expanduser().resolve() if quickdraw_dir else folder / "raw"
        manifests = {}
        for split in ("train", "test"):
            local = source / f"quickdraw_{split}.txt"
            manifests[split] = local if local.is_file() else download(QUICKDRAW_MANIFEST_URL.format(split=split), folder / f"downloads/quickdraw_{split}.txt", sha256=QUICKDRAW_MANIFEST_SHA256[split])
            if sha256_file(manifests[split]) != QUICKDRAW_MANIFEST_SHA256[split]:
                raise ValueError(f"{split} manifest does not match the verified official DomainNet QuickDraw manifest")
        splits, class_maps = {}, {}
        for split, manifest in manifests.items():
            splits[split], class_maps[split] = quickdraw_records(manifest)
        if class_maps["train"] != class_maps["test"] or set(class_maps["train"]) != set(range(345)):
            raise ValueError("Official DomainNet QuickDraw requires the same 345 indexed classes in both manifests")
        if len(splits["train"]) != 120750 or len(splits["test"]) != 51750:
            raise ValueError("Expected complete official QuickDraw splits (120750 train / 51750 test)")
        if not (source / "quickdraw").is_dir():
            if quickdraw_dir:
                raise FileNotFoundError(f"Expected quickdraw/ under {source}")
            archive = download(QUICKDRAW_URL, folder / "downloads/quickdraw.zip")
            safe_extract(archive, source)
        elif not quickdraw_dir and (folder / "downloads/quickdraw.zip").exists():
            safe_extract(folder / "downloads/quickdraw.zip", source)
        return write_manifests(folder, splits, [class_maps["train"][i] for i in range(345)], False,
                               image_roots={"train": source, "test": source},
                               provenance={"dataset": "DomainNet QuickDraw cleaned", "url": QUICKDRAW_URL, "manifests": {s: {"url": QUICKDRAW_MANIFEST_URL.format(split=s), "sha256": sha256_file(p)} for s, p in manifests.items()}})
    train_root = Path(chexpert_dir).expanduser() if chexpert_dir else folder / "raw/CheXpert-v1.0-small"
    test_root = Path(chexlocalize_dir).expanduser() if chexlocalize_dir else folder / "raw/CheXlocalize/CheXpert"
    if not (train_root / "train.csv").is_file() or not (test_root / "test_labels.csv").is_file():
        raise AccessRequired(CHEXPERT_ACCESS)
    splits = {"train": chexpert_records(train_root / "train.csv", train_root, "train"),
              "test": chexpert_records(test_root / "test_labels.csv", test_root, "test")}
    return write_manifests(folder, splits, CHEXPERT_LABELS, True,
                           image_roots={"train": train_root, "test": test_root},
                           provenance={"train": "CheXpert train.csv", "test": "CheXlocalize CheXpert/test_labels.csv", "label_policy": "uncertain and unreported labels ignored (-1)", "train_csv_sha256": sha256_file(train_root / "train.csv"), "test_csv_sha256": sha256_file(test_root / "test_labels.csv")})


def validate_model_checkpoint(path, model=DEFAULT_MODEL):
    """Check safetensors keys/shapes without allocating even the 7B weights."""
    import timm
    import torch
    from safetensors import safe_open

    spec = get_model_spec(model)
    with torch.device("meta"):
        architecture = timm.create_model(spec.timm_name, pretrained=False, num_classes=0)
    expected = {key: tuple(value.shape) for key, value in architecture.state_dict().items()}
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        actual = {key: tuple(checkpoint.get_slice(key).get_shape()) for key in checkpoint.keys()}
    missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
    mismatched = [key for key in expected.keys() & actual.keys() if expected[key] != actual[key]]
    if missing or extra or mismatched:
        raise ValueError(f"Checkpoint architecture mismatch: missing={missing[:8]}, unexpected={extra[:8]}, wrong_shapes={mismatched[:8]}")
    return {"model": spec.name, "hf_repo": spec.hf_repo, "timm_name": spec.timm_name,
            "tensors": len(actual), "parameters": sum(value.numel() for value in architecture.parameters()),
            "patch_size": spec.patch_size, "depth": spec.depth, "embed_dim": spec.embed_dim,
            "num_prefix_tokens": spec.num_prefix_tokens, "num_register_tokens": spec.num_register_tokens}


def prepare_model(destination=None, *, model=DEFAULT_MODEL, check_only=False):
    """Prepare one pinned official artifact; all registry revisions are single files."""
    spec = get_model_spec(model)
    destination = Path(destination or spec.default_checkpoint)
    if not check_only:
        download(spec.url, destination, sha256=spec.sha256, expected_size=spec.size)
    elif not destination.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {destination}")
    # download() already checks both on new transfers AND existing cached files.
    if check_only and (destination.stat().st_size != spec.size or sha256_file(destination) != spec.sha256):
        raise ValueError(f"SHA256 or size mismatch: checkpoint is not the pinned official {spec.name} artifact")
    details = validate_model_checkpoint(destination, model=spec)
    details.update({"revision": spec.revision, "sha256": spec.sha256, "url": spec.url,
                    "bytes": spec.size, "checkpoint": str(destination.resolve())})
    if not check_only:
        _atomic_json(destination.with_suffix(".json"), details)
    return details
