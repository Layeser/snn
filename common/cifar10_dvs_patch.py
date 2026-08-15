"""
Correctifs SpikingJelly CIFAR-10-DVS (compatibilité NumPy 2.x).

SpikingJelly utilise np.fromstring(..., dtype='>u4'), supprimé en NumPy 2.
Sans ce patch, la conversion .aedat → .npz échoue silencieusement dans les
threads et laisse des dossiers events_np/frames vides.
"""

from __future__ import annotations

import fcntl
import shutil
from pathlib import Path

import numpy as np

# CIFAR-10-DVS : 10 classes × 1000 séquences. Un cache partiel (course parallèle)
# ne doit jamais être considéré comme prêt.
CIFAR10_DVS_EXPECTED_SAMPLES = 10_000

_PATCHED = False


def apply_cifar10_dvs_numpy2_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import spikingjelly.datasets.cifar10_dvs as cifar10_dvs_module

    def load_raw_events(
        fp,
        bytes_skip=0,
        bytes_trim=0,
        filter_dvs=False,
        times_first=False,
    ):
        p = cifar10_dvs_module.skip_header(fp)
        fp.seek(p + bytes_skip)
        data = fp.read()
        if bytes_trim > 0:
            data = data[:-bytes_trim]
        data = np.frombuffer(data, dtype=">u4")
        if len(data) % 2 != 0:
            raise ValueError("odd number of data elements")
        raw_addr = data[::2]
        timestamp = data[1::2]
        if times_first:
            timestamp, raw_addr = raw_addr, timestamp
        if filter_dvs:
            valid = (
                cifar10_dvs_module.read_bits(
                    raw_addr,
                    cifar10_dvs_module.valid_mask,
                    cifar10_dvs_module.valid_shift,
                )
                == cifar10_dvs_module.EVT_DVS
            )
            timestamp = timestamp[valid]
            raw_addr = raw_addr[valid]
        return timestamp, raw_addr

    cifar10_dvs_module.load_raw_events = load_raw_events
    _PATCHED = True


def frames_dir(dvs_root: Path, frames_number: int, split_by: str = "number") -> Path:
    return dvs_root / f"frames_number_{frames_number}_split_by_{split_by}"


def count_npz_files(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.rglob("*.npz"))


def is_cifar10_dvs_frames_ready(
    dvs_root: Path,
    frames_number: int,
    split_by: str = "number",
) -> bool:
    return count_npz_files(frames_dir(dvs_root, frames_number, split_by)) >= CIFAR10_DVS_EXPECTED_SAMPLES


def reset_incomplete_cifar10_dvs_cache(
    dvs_root: Path,
    frames_number: int,
    split_by: str = "number",
    verbose: bool = True,
) -> None:
    """Supprime les caches incomplets (course parallèle, conversion interrompue)."""
    events_root = dvs_root / "events_np"
    frames_root = frames_dir(dvs_root, frames_number, split_by)

    n_events = count_npz_files(events_root)
    if events_root.is_dir() and n_events < CIFAR10_DVS_EXPECTED_SAMPLES:
        if verbose:
            print(f"Cache events_np incomplet ({n_events}/{CIFAR10_DVS_EXPECTED_SAMPLES}) supprimé → {events_root}")
        shutil.rmtree(events_root)

    n_frames = count_npz_files(frames_root)
    if frames_root.is_dir() and n_frames < CIFAR10_DVS_EXPECTED_SAMPLES:
        if verbose:
            print(f"Cache frames incomplet ({n_frames}/{CIFAR10_DVS_EXPECTED_SAMPLES}) supprimé → {frames_root}")
        shutil.rmtree(frames_root)

    for split_cache in dvs_root.glob(f"split_*_frames_{frames_number}.pt"):
        if verbose:
            print(f"Split cache DVS obsolète supprimé → {split_cache}")
        split_cache.unlink(missing_ok=True)


def prepare_cifar10_dvs_frames(
    dvs_root: Path,
    frames_number: int,
    split_by: str = "number",
    verbose: bool = True,
) -> None:
    """
    Télécharge (si besoin), convertit events_np et intègre les frames.
    Peut prendre 10–30 min au premier lancement.
    """
    apply_cifar10_dvs_numpy2_patch()
    dvs_root = Path(dvs_root)
    dvs_root.mkdir(parents=True, exist_ok=True)

    lock_path = dvs_root / ".prepare.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        _prepare_cifar10_dvs_frames_locked(
            dvs_root, frames_number, split_by, verbose=verbose
        )


def _prepare_cifar10_dvs_frames_locked(
    dvs_root: Path,
    frames_number: int,
    split_by: str,
    verbose: bool,
) -> None:
    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

    from data_download import download_cifar10_dvs_archives

    if is_cifar10_dvs_frames_ready(dvs_root, frames_number, split_by):
        if verbose:
            n = count_npz_files(frames_dir(dvs_root, frames_number, split_by))
            print(f"CIFAR-10-DVS frames prêtes ({n} fichiers) → {dvs_root}")
        return

    reset_incomplete_cifar10_dvs_cache(dvs_root, frames_number, split_by, verbose=verbose)
    download_cifar10_dvs_archives(dvs_root, verbose=verbose)

    if verbose:
        print(
            f"Préparation CIFAR-10-DVS (frames_number={frames_number}, "
            f"split_by={split_by}) — peut prendre plusieurs minutes…"
        )

    CIFAR10DVS(
        root=str(dvs_root),
        data_type="frame",
        frames_number=frames_number,
        split_by=split_by,
    )

    if not is_cifar10_dvs_frames_ready(dvs_root, frames_number, split_by):
        raise RuntimeError(
            f"La préparation CIFAR-10-DVS a échoué "
            f"({count_npz_files(frames_dir(dvs_root, frames_number, split_by))}"
            f"/{CIFAR10_DVS_EXPECTED_SAMPLES} frames). "
            "Vérifiez les archives dans download/ et relancez."
        )

    if verbose:
        n = count_npz_files(frames_dir(dvs_root, frames_number, split_by))
        print(f"CIFAR-10-DVS prêt ({n} séquences frame) → {dvs_root}")
