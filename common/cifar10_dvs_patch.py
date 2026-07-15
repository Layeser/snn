"""
Correctifs SpikingJelly CIFAR-10-DVS (compatibilité NumPy 2.x).

SpikingJelly utilise np.fromstring(..., dtype='>u4'), supprimé en NumPy 2.
Sans ce patch, la conversion .aedat → .npz échoue silencieusement dans les
threads et laisse des dossiers events_np/frames vides.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

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
    return count_npz_files(frames_dir(dvs_root, frames_number, split_by)) > 0


def reset_incomplete_cifar10_dvs_cache(
    dvs_root: Path,
    frames_number: int,
    split_by: str = "number",
    verbose: bool = True,
) -> None:
    """Supprime les caches vides laissés par une conversion échouée."""
    events_root = dvs_root / "events_np"
    frames_root = frames_dir(dvs_root, frames_number, split_by)

    if events_root.is_dir() and count_npz_files(events_root) == 0:
        if verbose:
            print(f"Cache DVS incomplet supprimé → {events_root}")
        shutil.rmtree(events_root)

    if frames_root.is_dir() and count_npz_files(frames_root) == 0:
        if verbose:
            print(f"Cache frames incomplet supprimé → {frames_root}")
        shutil.rmtree(frames_root)


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
    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

    from data_download import download_cifar10_dvs_archives

    apply_cifar10_dvs_numpy2_patch()
    dvs_root = Path(dvs_root)
    dvs_root.mkdir(parents=True, exist_ok=True)

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
            "La préparation CIFAR-10-DVS a échoué (aucun fichier frame .npz). "
            "Vérifiez les archives dans download/ et relancez."
        )

    if verbose:
        n = count_npz_files(frames_dir(dvs_root, frames_number, split_by))
        print(f"CIFAR-10-DVS prêt ({n} séquences frame) → {dvs_root}")
