"""download_tinyimagenet: unpacks the archive once, falls back to the next source, and discards bad archives (no network)."""
import zipfile

import pytest

from multigeom.data import download_tinyimagenet


def make_archive(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("tiny-imagenet-200/train/n01/images/a.JPEG", b"x")
        z.writestr("tiny-imagenet-200/val/images/b.JPEG", b"x")
    return path


def test_unpacks_and_is_idempotent(tmp_path):
    archive = make_archive(tmp_path / "src.zip")
    root = download_tinyimagenet(tmp_path / "data", urls=(archive.as_uri(),), expected_bytes=archive.stat().st_size)
    assert (root / "train" / "n01" / "images" / "a.JPEG").exists() and (root / "val" / "images" / "b.JPEG").exists()
    assert not (tmp_path / "data" / "tiny-imagenet-200.zip").exists()
    assert download_tinyimagenet(tmp_path / "data", urls=()) == root       # already there: nothing is downloaded


def test_falls_back_to_the_next_source(tmp_path):
    archive = make_archive(tmp_path / "src.zip")
    root = download_tinyimagenet(tmp_path / "data", urls=((tmp_path / "missing.zip").as_uri(), archive.as_uri()), expected_bytes=None)
    assert (root / "val" / "images" / "b.JPEG").exists()


def test_wrong_size_is_rejected(tmp_path):
    archive = make_archive(tmp_path / "src.zip")
    with pytest.raises(RuntimeError, match="expected"):
        download_tinyimagenet(tmp_path / "data", urls=(archive.as_uri(),), expected_bytes=1)
    assert not (tmp_path / "data" / "tiny-imagenet-200").exists() and not (tmp_path / "data" / "tiny-imagenet-200.zip").exists()
