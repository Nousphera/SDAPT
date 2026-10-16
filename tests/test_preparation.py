import csv
import hashlib
import io
import json
import tarfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from sdapt.data import CHEXPERT_LABELS
from sdapt.preparation import (
    chexpert_records, download, fmow_records, quickdraw_records,
    safe_extract, write_manifests,
    prepare_dataset, validate_model_checkpoint,
)


def test_fmow_uses_wilds_official_time_and_original_splits(tmp_path):
    path = tmp_path / "rgb_metadata.csv"
    path.write_text("category,split,timestamp\nairport,train,2012-12-31T23:59:59Z\nairport,train,2013-01-01T00:00:00Z\nairport,test,2015-12-31T23:59:59Z\nairport,test,2016-01-01T00:00:00Z\nairport,seq,2017-01-01T00:00:00Z\n")
    records = fmow_records(path)
    assert [r["path"] for r in records["train"]] == ["images/rgb_img_0.png"]
    assert [r["path"] for r in records["test"]] == ["images/rgb_img_3.png"]


def test_quickdraw_class_mapping_and_duplicate_rejection(tmp_path):
    path = tmp_path / "quickdraw_train.txt"
    path.write_text("quickdraw/apple/a.png 0\nquickdraw/bird/b.png 1\n")
    rows, classes = quickdraw_records(path)
    assert classes == {0: "apple", 1: "bird"}
    assert [row["target"] for row in rows] == [0, 1]
    path.write_text("quickdraw/apple/a.png 0\nquickdraw/apple/a.png 0\n")
    with pytest.raises(ValueError, match="duplicate"):
        quickdraw_records(path)


def test_chexpert_unknown_labels_are_masked_and_columns_required(tmp_path):
    source = tmp_path / "train.csv"
    image = tmp_path / "train/patient1/study1/view1.png"
    image.parent.mkdir(parents=True)
    Image.new("L", (8, 8)).save(image)
    with source.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Path", *CHEXPERT_LABELS])
        writer.writerow(["CheXpert-v1.0-small/train/patient1/study1/view1.png", "1", "-1", "", "0", "0.0", "1.0", "", "0"])
    rows = chexpert_records(source, tmp_path, "train")
    assert rows[0]["target"] == [1, -1, -1, 0, 0, 1, -1, 0]
    assert rows[0]["patient"] == "patient1"
    source.write_text("Path,Edema\ntrain/patient1/study1/view1.png,1\n")
    with pytest.raises(ValueError, match="columns"):
        chexpert_records(source, tmp_path, "train")


def test_rejects_train_test_image_and_patient_leakage(tmp_path):
    Image.new("L", (8, 8)).save(tmp_path / "a.png")
    Image.new("L", (8, 8)).save(tmp_path / "b.png")
    with pytest.raises(ValueError, match="overlap"):
        write_manifests(tmp_path, {"train": [{"path": "a.png", "target": 0}], "test": [{"path": "a.png", "target": 0}]}, ["a"], False)
    with pytest.raises(ValueError, match="patient"):
        write_manifests(tmp_path, {"train": [{"path": "a.png", "target": 0, "patient": "p1"}], "test": [{"path": "b.png", "target": 0, "patient": "p1"}]}, ["a"], False)
    assert not (tmp_path / "metadata.json").exists()


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_archive_traversal_is_rejected_before_extraction(tmp_path, kind):
    archive = tmp_path / ("bad." + kind)
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as file:
            file.writestr("safe.txt", "fine")
            file.writestr("../escape", "bad")
    else:
        with tarfile.open(archive, "w") as file:
            for name in ["safe.txt", "../escape"]:
                info = tarfile.TarInfo(name)
                info.size = 1
                file.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="path"):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "out/safe.txt").exists()


def test_valid_archive_extracts_idempotently(tmp_path):
    archive = tmp_path / "good.zip"
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr("images/a.txt", "abc")
    safe_extract(archive, tmp_path / "out")
    safe_extract(archive, tmp_path / "out")
    assert (tmp_path / "out/images/a.txt").read_text() == "abc"


@pytest.mark.parametrize("support_range", [True, False])
def test_resume_download_and_servers_ignoring_range(tmp_path, support_range):
    payload = b"verified payload" * 100
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0]) if support_range else 0
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(payload) - start))
            self.send_header("ETag", '"fixture-v1"')
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(payload)-1}/{len(payload)}")
            self.end_headers()
            self.wfile.write(payload[start:])
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = tmp_path / "download.bin"
    target.with_suffix(".bin.part").write_bytes(payload[:37])
    try:
        download(f"http://127.0.0.1:{server.server_port}/file", target, sha256=hashlib.sha256(payload).hexdigest())
    finally:
        server.shutdown()
        server.server_close()
    assert target.read_bytes() == payload
    assert not target.with_suffix(".bin.part").exists()


def test_download_hash_mismatch_does_not_publish(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA256"):
        download(source.as_uri(), tmp_path / "target", sha256="0" * 64)
    assert not (tmp_path / "target").exists()


def test_local_fmow_preparation_round_trip(tmp_path):
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "rgb_metadata.csv").write_text("category,split,timestamp\nairport,train,2012-01-01T00:00:00Z\nwind_farm,test,2016-01-01T00:00:00Z\n")
    for i in range(2):
        Image.new("RGB", (16, 16)).save(source / f"images/rgb_img_{i}.png")
    metadata = prepare_dataset("fmow", tmp_path / "data", fmow_dir=source)
    assert metadata["counts"] == {"train": 1, "test": 1}
    assert prepare_dataset("fmow", tmp_path / "data", check_only=True) == metadata


def test_checkpoint_validation_rejects_wrong_architecture(tmp_path):
    import torch
    from safetensors.torch import save_file
    path = tmp_path / "wrong.safetensors"
    save_file({"cls_token": torch.zeros(1, 1, 2)}, path)
    with pytest.raises(ValueError, match="architecture mismatch"):
        validate_model_checkpoint(path)


def test_registry_names_resolve_and_unknown_models_fail():
    from sdapt.registry import get_model_spec
    assert get_model_spec("dinov2-base").name == "dinov2-vitb14"
    assert get_model_spec("dinov3-vit7b16").embed_dim == 4096
    assert get_model_spec("dinov3-vitl16-sat493m").num_register_tokens == 4
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_spec("made-up-model")


def test_registry_geometry_matches_instantiated_architectures_without_allocating_weights():
    import timm
    import torch
    from sdapt.registry import MODEL_SPECS
    for spec in MODEL_SPECS.values():
        with torch.device("meta"):
            model = timm.create_model(spec.timm_name, pretrained=False, num_classes=0)
        assert len(model.blocks) == spec.depth
        assert model.patch_embed.patch_size == (spec.patch_size, spec.patch_size)
        assert model.embed_dim == spec.embed_dim
        assert model.num_prefix_tokens == spec.num_prefix_tokens
        assert all(tensor.device.type == "meta" for tensor in model.parameters())


def test_known_large_download_checks_disk_before_contacting_source(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sdapt.preparation as preparation
    monkeypatch.setattr(preparation.shutil, "disk_usage", lambda path: SimpleNamespace(free=10))
    # This URL is deliberately unreachable: low space must fail before networking.
    with pytest.raises(OSError, match="Insufficient free disk"):
        download("http://127.0.0.1:1/unreachable", tmp_path / "large.safetensors", expected_size=26_864_183_416)
    assert not (tmp_path / "large.safetensors.part").exists()


def test_verified_cached_weights_are_reused_even_when_disk_is_full(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sdapt.preparation as preparation
    cached = tmp_path / "cached.safetensors"
    cached.write_bytes(b"cached artifact")
    monkeypatch.setattr(preparation.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    result = download("http://127.0.0.1:1/unreachable", cached, expected_size=15,
                      sha256=hashlib.sha256(b"cached artifact").hexdigest())
    assert result.read_bytes() == b"cached artifact"
