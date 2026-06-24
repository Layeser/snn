"""
Téléchargement accéléré des jeux de données.

CIFAR-10 : miroirs alternatifs (le serveur Toronto est souvent très lent).
CIFAR-10-DVS : téléchargement parallèle des 10 archives Figshare.

Usage :
    python scripts/download_data.py cifar10
    python scripts/download_data.py cifar10-dvs
    python scripts/download_data.py all
"""

from __future__ import annotations

import hashlib
import shutil
import ssl
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from tqdm import tqdm

CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_EXTRACTED_DIR = "cifar-10-batches-py"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_MIRRORS = [
  # ~30x plus rapide que le serveur Toronto depuis l'Europe
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]

CHUNK_SIZE = 1024 * 1024  # 1 MiB (torchvision utilise 8 KiB par défaut)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def is_cifar10_ready(data_dir: Path) -> bool:
    return (data_dir / CIFAR10_EXTRACTED_DIR / "batches.meta").exists()


def is_cifar10_dvs_archives_ready(dvs_root: Path) -> bool:
    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

    download_root = dvs_root / "download"
    if not download_root.is_dir():
        return False
    for file_name, _url, expected_md5 in CIFAR10DVS.resource_url_md5():
        fpath = download_root / file_name
        if not fpath.is_file() or md5_file(fpath) != expected_md5:
            return False
    return True


def _try_aria2c(url: str, dest: Path) -> bool:
    if shutil.which("aria2c") is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aria2c",
            "-x",
            "16",
            "-s",
            "16",
            "-k",
            "1M",
            "-d",
            str(dest.parent),
            "-o",
            dest.name,
            "--continue=true",
            "--allow-overwrite=true",
            url,
        ],
        check=True,
    )
    return dest.is_file()


def _download_urllib(url: str, dest: Path, desc: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "snn-data-download/1.0"})
    ctx = ssl.create_default_context()
    with urlopen(req, context=ctx, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0))
        label = desc or dest.name
        with dest.open("wb") as out, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=label,
            miniters=1,
        ) as bar:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))


def download_file(url: str, dest: Path, md5: str | None = None, desc: str | None = None) -> None:
    if dest.is_file() and md5 is not None and md5_file(dest) == md5:
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.is_file():
        tmp.unlink()

    try:
        if not _try_aria2c(url, dest):
            _download_urllib(url, dest, desc=desc)
    except (URLError, subprocess.CalledProcessError, OSError) as exc:
        if dest.is_file():
            dest.unlink(missing_ok=True)
        raise RuntimeError(f"Échec du téléchargement depuis {url}") from exc

    if md5 is not None and md5_file(dest) != md5:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 invalide pour {dest.name} (attendu {md5})")


def _extract_cifar10(archive: Path, data_dir: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=data_dir)


def ensure_cifar10(data_dir: Path, verbose: bool = True) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if is_cifar10_ready(data_dir):
        if verbose:
            print(f"CIFAR-10 déjà présent → {data_dir / CIFAR10_EXTRACTED_DIR}")
        return

    archive = data_dir / CIFAR10_ARCHIVE
    if archive.is_file() and md5_file(archive) == CIFAR10_MD5:
        if verbose:
            print(f"Archive trouvée, extraction → {archive}")
        _extract_cifar10(archive, data_dir)
        if is_cifar10_ready(data_dir):
            return

    errors: list[str] = []
    for url in CIFAR10_MIRRORS:
        if verbose:
            print(f"Téléchargement CIFAR-10 depuis {url}")
        try:
            download_file(url, archive, md5=CIFAR10_MD5, desc="CIFAR-10")
            _extract_cifar10(archive, data_dir)
            if verbose:
                print(f"CIFAR-10 prêt → {data_dir / CIFAR10_EXTRACTED_DIR}")
            return
        except RuntimeError as exc:
            errors.append(f"{url}: {exc}")
            archive.unlink(missing_ok=True)

    raise RuntimeError("Impossible de télécharger CIFAR-10.\n" + "\n".join(errors))


def download_cifar10_dvs_archives(dvs_root: Path, max_workers: int = 4, verbose: bool = True) -> None:
    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

    dvs_root = Path(dvs_root)
    download_root = dvs_root / "download"
    download_root.mkdir(parents=True, exist_ok=True)
    resources = CIFAR10DVS.resource_url_md5()

    if is_cifar10_dvs_archives_ready(dvs_root):
        if verbose:
            print(f"Archives CIFAR-10-DVS déjà présentes → {download_root}")
        return

    pending = [
        item for item in resources if not (download_root / item[0]).is_file()
        or md5_file(download_root / item[0]) != item[2]
    ]
    if not pending and is_cifar10_dvs_archives_ready(dvs_root):
        return

    if verbose:
        print(f"Téléchargement parallèle de {len(pending)} archives CIFAR-10-DVS…")

    def _download_one(item: tuple[str, str, str]) -> str:
        file_name, url, expected_md5 = item
        dest = download_root / file_name
        if dest.is_file() and md5_file(dest) == expected_md5:
            return file_name
        download_file(url, dest, md5=expected_md5, desc=file_name)
        return file_name

    workers = min(max_workers, max(1, len(pending)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, item): item[0] for item in pending}
        for future in as_completed(futures):
            name = futures[future]
            future.result()
            if verbose:
                print(f"  ✓ {name}")

    if verbose:
        print(f"Archives CIFAR-10-DVS prêtes → {download_root}")
        print("Note: la conversion events_np au premier entraînement peut prendre plusieurs minutes.")
