"""download_tvsd and download_things_images: local files stand for the servers (no network)."""
import shutil
import subprocess
import zipfile

import pytest

from multigeom.data import download_things_images, download_tvsd
from multigeom.data.download import download_file

LICENCE = "I agree to the licence.\n\nPassword for images_THINGS.zip: secret1\n"


def test_download_file_resumes_and_checks_the_size(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(bytes(range(256)) * 40)
    dest = tmp_path / "out" / "file.bin"
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "file.bin.part").write_bytes(source.read_bytes()[:100])    # an interrupted download
    with pytest.raises(OSError):     # file:// does not support Range: the whole file is downloaded again
        download_file(source.as_uri(), dest, expected_bytes=1)
    assert not dest.exists() and not dest.with_name("file.bin.part").exists()
    download_file(source.as_uri(), dest, expected_bytes=source.stat().st_size)
    assert dest.read_bytes() == source.read_bytes()


def test_download_tvsd_skips_what_is_there(tmp_path):
    source = tmp_path / "a.mat"
    source.write_bytes(b"x" * 10)
    files = {"monkeyN_a.mat": (source.as_uri(), 10)}
    root = download_tvsd(tmp_path / "data", files=files)
    assert (root / "monkeyN_a.mat").read_bytes() == b"x" * 10
    files = {"monkeyN_a.mat": ((tmp_path / "gone.mat").as_uri(), 10)}       # no source any more: nothing is downloaded
    assert download_tvsd(tmp_path / "data", files=files) == root


def make_things_archive(tmp_path):
    images = tmp_path / "build" / "object_images" / "aardvark"
    images.mkdir(parents=True)
    (images / "aardvark_01b.jpg").write_bytes(b"jpg")
    archive = tmp_path / "images_THINGS.zip"
    subprocess.run(["zip", "-q", "-r", "-P", "secret1", str(archive), "object_images"], cwd=tmp_path / "build", check=True)
    password = tmp_path / "password.txt"
    password.write_text(LICENCE)
    return archive, password


def test_things_images_needs_the_licence_to_be_accepted(tmp_path):
    archive, password = make_things_archive(tmp_path) if shutil.which("zip") else (None, None)
    if archive is None:
        pytest.skip("zip is not installed")
    with pytest.raises(ValueError, match="accept_license=True"):
        download_things_images(tmp_path / "data", url=archive.as_uri(), expected_bytes=None, password_url=password.as_uri())
    assert not (tmp_path / "data" / "things_images" / "images_THINGS.zip").exists()       # nothing was downloaded

    root = download_things_images(tmp_path / "data", accept_license=True, url=archive.as_uri(), expected_bytes=None,
                                  password_url=password.as_uri())
    assert (root / "object_images" / "aardvark" / "aardvark_01b.jpg").read_bytes() == b"jpg"
    assert not (root / "images_THINGS.zip").exists() and not (root / "unpacking").exists()
    assert download_things_images(tmp_path / "data", url="file:///gone") == root        # already there


def test_things_images_without_the_unzip_program(tmp_path, monkeypatch):
    if not shutil.which("zip"):
        pytest.skip("zip is not installed")
    archive, password = make_things_archive(tmp_path)
    monkeypatch.setattr("multigeom.data.download.shutil.which", lambda name: None)       # the Python decryption is used
    root = download_things_images(tmp_path / "data", accept_license=True, url=archive.as_uri(), expected_bytes=None,
                                  password_url=password.as_uri())
    assert (root / "object_images" / "aardvark" / "aardvark_01b.jpg").read_bytes() == b"jpg"
