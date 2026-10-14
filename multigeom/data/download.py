"""
Download the datasets of the paper, from a notebook or a script: `download_tinyimagenet`, `download_tvsd` and
`download_things_images`. Each one does nothing if its folder is already under `data_dir()` (see `multigeom.config`).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from ..config import data_dir

TVSD_FILES = {
    # file name in data/tvsd/: (URL, size in bytes). The URL must contain /raw/: /src/ returns the web page of the file.
    "monkeyN_THINGS_normMUA.mat": ("https://gin.g-node.org/doi/TVSD/raw/master/monkeyN/THINGS_normMUA.mat", 204465522),
    "things_imgs.mat": ("https://gin.g-node.org/doi/TVSD/raw/master/monkeyN/_logs/things_imgs.mat", 32769935),
}
THINGS_IMAGES_URL = "https://osf.io/download/rdxy2/"          # images_THINGS.zip (5.0 GB), https://osf.io/jum2f/
THINGS_IMAGES_BYTES = 5022473688
THINGS_PASSWORD_URL = "https://osf.io/download/j6a3m/"        # a licence text and the password of the archive
# Primary source first, then a mirror serving the identical archive.
TINYIMAGENET_URLS = (
    "https://cs231n.stanford.edu/tiny-imagenet-200.zip",
    "https://image-net.org/data/tiny-imagenet-200.zip",
)
TINYIMAGENET_BYTES = 248100043      # size of the archive (240 MB; 480 MB once unpacked)


def download_file(url: str, dest: Path, expected_bytes: int | None = None, chunk: int = 1 << 20) -> Path:
    """
    Download `url` to `dest`. An interrupted download is resumed (from `dest` + ".part") when the server allows it. The file is
    only moved to `dest` once it has the expected size, so `dest` is never a truncated file.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(request) as response:
        resumed = start > 0 and response.status == 206
        with open(part, "ab" if resumed else "wb") as out:
            shutil.copyfileobj(response, out, chunk)
    size = part.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        part.unlink()
        raise OSError(f"{url}: {size} bytes, expected {expected_bytes}")
    part.replace(dest)
    return dest


def download_tinyimagenet(
    data_directory: str | Path | None = None,
    urls: tuple[str, ...] = TINYIMAGENET_URLS,
    expected_bytes: int | None = TINYIMAGENET_BYTES,
) -> Path:
    """
    Download and unpack TinyImageNet into `<data_directory>/tiny-imagenet-200` (default: `data_dir()`), and return that folder.
    Nothing is done if the folder is already there. The sources are tried in order; an archive of the wrong size, or that
    is not a valid zip, is discarded. About 1 minute.

    Args:
        data_directory: the folder that will contain `tiny-imagenet-200`.
        urls: where to download the archive from, in order.
        expected_bytes: the size of the archive, to detect a truncated download (`None` skips the check).
    """
    data_directory = Path(data_directory) if data_directory else data_dir()
    root = data_directory / "tiny-imagenet-200"
    if (root / "train").is_dir() and (root / "val" / "images").is_dir():
        return root

    archive = data_directory / "tiny-imagenet-200.zip"
    errors = []
    for url in urls:
        try:
            print(f"downloading {url}")
            archive.unlink(missing_ok=True)
            download_file(url, archive, expected_bytes)
            with zipfile.ZipFile(archive) as z:
                if z.testzip() is not None:
                    raise OSError("corrupt archive")
                z.extractall(data_directory)
            break
        except (OSError, zipfile.BadZipFile) as e:      # URLError is an OSError
            errors.append(f"{url}: {e}")
            shutil.rmtree(root, ignore_errors=True)     # a partial unpacking
        finally:
            archive.unlink(missing_ok=True)
    else:
        raise RuntimeError("could not download TinyImageNet:\n  " + "\n  ".join(errors))
    return root


def download_tvsd(data_directory: str | Path | None = None, files: dict | None = None) -> Path:
    """
    Download the THINGS Ventral-stream Spiking Dataset files used here (monkey N: 205 MB and 33 MB) into
    `<data_directory>/tvsd` (default: `data_dir()`), from GIN, and return that folder. Files already there are kept.
    """
    root = (Path(data_directory) if data_directory else data_dir()) / "tvsd"
    for name, (url, nbytes) in (files or TVSD_FILES).items():
        path = root / name
        if path.exists() and path.stat().st_size == nbytes:
            continue
        print(f"downloading {url}")
        download_file(url, path, nbytes)
    return root


def download_things_images(
    data_directory: str | Path | None = None,
    accept_license: bool = False,
    url: str = THINGS_IMAGES_URL,
    expected_bytes: int | None = THINGS_IMAGES_BYTES,
    password_url: str = THINGS_PASSWORD_URL,
) -> Path:
    """
    Download the THINGS object images (5.0 GB, password-protected archive on OSF) and unpack them into
    `<data_directory>/things_images/object_images/<concept>/<image>.jpg` (default: `data_dir()`). Returns the folder
    `things_images`. It takes about 20 minutes, and the interrupted download resumes.

    The images are shared under a licence: research and non-commercial use only, no redistribution or alteration. The password
    is published together with this licence, so you must read it and pass `accept_license=True`.
    """
    root = (Path(data_directory) if data_directory else data_dir()) / "things_images"
    if (root / "object_images").is_dir():
        return root

    with urllib.request.urlopen(password_url) as response:
        text = response.read().decode("utf-8", errors="replace")
    match = re.search(r"zip:\s*(\S+)", text)
    if match is None:
        raise RuntimeError(f"could not read the password of the archive from {password_url}")
    if not accept_license:
        raise ValueError(f"{text.strip()}\n\nRead the licence above (https://osf.io/jum2f/) and call "
                         "download_things_images(accept_license=True) if you agree with it.")
    password = match.group(1)

    archive = root / "images_THINGS.zip"
    if not archive.exists():
        print(f"downloading {url} ({(expected_bytes or 0) / 1e9:.1f} GB)")
        download_file(url, archive, expected_bytes)
    staging = root / "unpacking"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    print("unpacking")
    if shutil.which("unzip"):                                   # fast; Python decrypts the archive at about 1 MB/s
        done = subprocess.run(["unzip", "-q", "-o", "-P", password, str(archive), "-d", str(staging)])
        if done.returncode != 0:
            raise RuntimeError(f"unzip failed on {archive} (remove it to download it again)")
    else:
        with zipfile.ZipFile(archive) as z:
            z.extractall(staging, pwd=password.encode())
    for child in staging.iterdir():
        child.replace(root / child.name)
    staging.rmdir()
    archive.unlink()
    return root
