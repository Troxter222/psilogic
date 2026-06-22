"""Dataset paths, local cache helpers and offline pre-download utilities.

Expected layout under ``--data-root`` (default ``./data``)::

    data/
    ├── tinystories/
    │   ├── train_tokens.npy    # uint8 byte tokens (benchmark subset)
    │   ├── val_tokens.npy
    │   └── meta.json
    ├── cifar-100-python/       # torchvision CIFAR-100
    ├── tiny-imagenet-200/      # Stanford Tiny ImageNet
    └── celeba/                 # torchvision OR HuggingFace (Dataset/CelebA_train/…)

Pre-download on a fast connection, copy the whole ``data/`` folder to RunPod /
Jupyter, then run::

    python -m fairbench --data-root /workspace/data --offline
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, Optional, Tuple

from .logging_utils import LOGGER

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

TINYSTORIES_SUBDIR = "tinystories"
CIFAR100_DIR = "cifar-100-python"
TINY_IMAGENET_DIR = "tiny-imagenet-200"
CELEBA_DIR = "celeba"
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

DEFAULT_TRAIN_CHARS = 2_000_000
DEFAULT_VAL_CHARS = 200_000


def tinystories_paths(data_root: str) -> dict[str, str]:
    base = os.path.join(data_root, TINYSTORIES_SUBDIR)
    return {
        "dir": base,
        "train": os.path.join(base, "train_tokens.npy"),
        "val": os.path.join(base, "val_tokens.npy"),
        "meta": os.path.join(base, "meta.json"),
    }


def tinystories_ready(data_root: str) -> bool:
    p = tinystories_paths(data_root)
    return os.path.isfile(p["train"]) and os.path.isfile(p["val"])


def cifar100_ready(data_root: str) -> bool:
    return os.path.isdir(os.path.join(data_root, CIFAR100_DIR))


def tiny_imagenet_ready(data_root: str) -> bool:
    root = os.path.join(data_root, TINY_IMAGENET_DIR)
    return os.path.isdir(os.path.join(root, "train")) and os.path.isdir(os.path.join(root, "val"))


def _dir_has_images(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for name in os.listdir(path):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            return True
    return False


def find_celeba_image_dirs(data_root: str) -> dict[str, str]:
    """Locate CelebA face image folders (torchvision or HuggingFace layout).

    Returns a mapping such as ``{"torchvision": "/.../img_align_celeba"}`` when
    the classic torchvision tree is present, or ``{"train": "...", "valid": "..."}``
    for the HuggingFace ``Dataset/CelebA_*`` layout.
    """
    base = os.path.join(data_root, CELEBA_DIR)
    if not os.path.isdir(base):
        return {}

    for rel in ("img_align_celeba", os.path.join("celeba", "img_align_celeba")):
        tv = os.path.join(base, rel)
        if _dir_has_images(tv):
            return {"torchvision": tv}

    hf_splits = {
        "train": os.path.join(base, "Dataset", "CelebA_train", "img_align_celeba"),
        "valid": os.path.join(base, "Dataset", "CelebA_valid", "img_align_celeba"),
        "test": os.path.join(base, "Dataset", "CelebA_test", "img_align_celeba"),
    }
    found = {k: v for k, v in hf_splits.items() if _dir_has_images(v)}
    if found:
        return found

    # Any other img_align_celeba tree (manual uploads).
    for dirpath, _, _ in os.walk(base):
        if os.path.basename(dirpath) == "img_align_celeba" and _dir_has_images(dirpath):
            return {"train": dirpath}
    return {}


def celeba_ready(data_root: str) -> bool:
    return bool(find_celeba_image_dirs(data_root))


def celeba_uses_torchvision(data_root: str) -> bool:
    return "torchvision" in find_celeba_image_dirs(data_root)


def dataset_status(data_root: str) -> dict[str, bool]:
    return {
        "tinystories": tinystories_ready(data_root),
        "cifar100": cifar100_ready(data_root),
        "tiny_imagenet": tiny_imagenet_ready(data_root),
        "celeba": celeba_ready(data_root),
    }


def require_local(name: str, ready: bool, data_root: str, offline: bool) -> None:
    """Raise a clear error in offline mode when a dataset is missing."""
    if ready or not offline:
        return
    raise FileNotFoundError(
        f"Offline mode: dataset '{name}' not found under {os.path.abspath(data_root)}.\n"
        f"Pre-download on your PC:\n"
        f"  python -m fairbench.download --data-root {data_root}\n"
        f"Then copy the entire data/ folder to the cloud machine and rerun with --offline."
    )


# --------------------------------------------------------------------------- #
# TinyStories cache (small, upload-friendly)
# --------------------------------------------------------------------------- #


def save_tinystories_cache(
    data_root: str,
    train_ids,
    val_ids,
    train_chars: int = DEFAULT_TRAIN_CHARS,
    val_chars: int = DEFAULT_VAL_CHARS,
) -> None:
    import numpy as np

    paths = tinystories_paths(data_root)
    os.makedirs(paths["dir"], exist_ok=True)
    np.save(paths["train"], np.asarray(train_ids, dtype=np.uint8))
    np.save(paths["val"], np.asarray(val_ids, dtype=np.uint8))
    meta = {"train_chars": train_chars, "val_chars": val_chars, "format": "uint8_bytes"}
    with open(paths["meta"], "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    LOGGER.info(
        "Saved TinyStories cache -> %s (train=%d, val=%d tokens)",
        paths["dir"],
        len(train_ids),
        len(val_ids),
    )


def load_tinystories_cache(data_root: str) -> tuple[Any, Any]:
    import numpy as np

    paths = tinystories_paths(data_root)
    train = np.load(paths["train"])
    val = np.load(paths["val"])
    LOGGER.info(
        "Loaded TinyStories from local cache (%s): %d train, %d val tokens.",
        paths["dir"],
        len(train),
        len(val),
    )
    return train, val


def reorganize_tiny_imagenet_val(val_dir: str) -> None:
    """Convert Tiny ImageNet flat val/ into per-class ImageFolder layout."""
    marker = os.path.join(val_dir, ".reorganized")
    if os.path.exists(marker):
        return
    ann = os.path.join(val_dir, "val_annotations.txt")
    img_dir = os.path.join(val_dir, "images")
    if not os.path.exists(ann) or not os.path.isdir(img_dir):
        return
    with open(ann, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            fname, wnid = parts[0], parts[1]
            cls_dir = os.path.join(val_dir, wnid)
            os.makedirs(cls_dir, exist_ok=True)
            src = os.path.join(img_dir, fname)
            if os.path.exists(src):
                shutil.move(src, os.path.join(cls_dir, fname))
    shutil.rmtree(img_dir, ignore_errors=True)
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write("done")


def fetch_tinystories_text(split: str, char_budget: int) -> str:
    """Fetch up to ``char_budget`` characters from TinyStories (train/validation).

    Tries HuggingFace ``datasets`` streaming first; falls back to the public
    HF Datasets Server API (stdlib only, no extra pip packages).
    """
    try:
        from datasets import load_dataset

        LOGGER.info("Streaming TinyStories/%s via HuggingFace datasets...", split)
        stream = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        buf, total = [], 0
        for ex in stream:
            txt = ex.get("text", "")
            buf.append(txt)
            total += len(txt)
            if total >= char_budget:
                break
        return "\n".join(buf)[:char_budget]
    except ImportError:
        LOGGER.info("Package 'datasets' not installed; using HuggingFace Datasets Server API.")
    except Exception as exc:
        LOGGER.warning("HuggingFace datasets streaming failed (%s); trying API fallback.", exc)

    return _fetch_tinystories_via_hf_api(split, char_budget)


def _fetch_tinystories_via_hf_api(split: str, char_budget: int) -> str:
    """Download TinyStories text via datasets-server.huggingface.co (stdlib only)."""
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    # HF split name for validation is "validation".
    hf_split = split if split != "val" else "validation"
    buf: list[str] = []
    total = 0
    offset = 0
    page = 100

    LOGGER.info("Fetching TinyStories/%s via HF API (budget=%d chars)...", hf_split, char_budget)
    while total < char_budget:
        params = urllib.parse.urlencode(
            {
                "dataset": "roneneldan/TinyStories",
                "config": "default",
                "split": hf_split,
                "offset": offset,
                "length": page,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{params}"
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not download TinyStories. Install HuggingFace datasets for a "
                "more reliable download:\n  pip install datasets\n"
                f"API error: {exc}"
            ) from exc

        rows = payload.get("rows") or []
        if not rows:
            break
        for row in rows:
            txt = (row.get("row") or {}).get("text", "")
            if not txt:
                continue
            buf.append(txt)
            total += len(txt)
            if total >= char_budget:
                break
        offset += page

    if not buf:
        raise RuntimeError(
            f"TinyStories/{hf_split} returned no rows from the HF API. Try: pip install datasets"
        )
    return "\n".join(buf)[:char_budget]


def download_tinystories(
    data_root: str,
    train_chars: int = DEFAULT_TRAIN_CHARS,
    val_chars: int = DEFAULT_VAL_CHARS,
    force: bool = False,
) -> None:
    """Stream TinyStories from HuggingFace and write a compact local cache."""
    if tinystories_ready(data_root) and not force:
        LOGGER.info("TinyStories cache already present; skipping.")
        return

    LOGGER.info(
        "Downloading TinyStories subset (train=%d, val=%d chars)...", train_chars, val_chars
    )
    train_text = fetch_tinystories_text("train", train_chars)
    val_text = fetch_tinystories_text("validation", val_chars)
    import numpy as np

    train_ids = np.frombuffer(train_text.encode("utf-8", "replace"), dtype=np.uint8)
    val_ids = np.frombuffer(val_text.encode("utf-8", "replace"), dtype=np.uint8)
    save_tinystories_cache(data_root, train_ids, val_ids, train_chars, val_chars)


def download_cifar100(data_root: str, force: bool = False) -> None:
    if cifar100_ready(data_root) and not force:
        LOGGER.info("CIFAR-100 already present; skipping.")
        return
    from torchvision.datasets import CIFAR100

    os.makedirs(data_root, exist_ok=True)
    LOGGER.info("Downloading CIFAR-100 (~169 MB) -> %s", data_root)
    CIFAR100(root=data_root, train=True, download=True)
    CIFAR100(root=data_root, train=False, download=True)


def download_tiny_imagenet(data_root: str, force: bool = False) -> None:
    if tiny_imagenet_ready(data_root) and not force:
        LOGGER.info("Tiny ImageNet already present; skipping.")
        return
    from torchvision.datasets.utils import download_and_extract_archive

    os.makedirs(data_root, exist_ok=True)
    LOGGER.info("Downloading Tiny ImageNet (~240 MB zip) -> %s", data_root)
    download_and_extract_archive(TINY_IMAGENET_URL, download_root=data_root)
    reorganize_tiny_imagenet_val(os.path.join(data_root, TINY_IMAGENET_DIR, "val"))


def download_celeba(data_root: str, force: bool = False) -> None:
    if celeba_ready(data_root) and not force:
        LOGGER.info("CelebA already present; skipping.")
        return
    from torchvision.datasets import CelebA

    os.makedirs(data_root, exist_ok=True)
    LOGGER.info("Downloading CelebA (~1.3 GB; may take a while) -> %s", data_root)
    for split in ("train", "valid", "test"):
        CelebA(root=data_root, split=split, target_type=[], download=True)


def download_all(
    data_root: str = "./data",
    train_chars: int = DEFAULT_TRAIN_CHARS,
    val_chars: int = DEFAULT_VAL_CHARS,
    skip_celeba: bool = False,
    force: bool = False,
) -> dict[str, bool]:
    """Download every benchmark dataset into ``data_root``."""
    os.makedirs(data_root, exist_ok=True)
    LOGGER.info("Dataset root: %s", os.path.abspath(data_root))

    download_tinystories(data_root, train_chars, val_chars, force=force)
    download_cifar100(data_root, force=force)
    download_tiny_imagenet(data_root, force=force)
    if skip_celeba:
        LOGGER.warning(
            "Skipping CelebA (--skip-celeba). Diffusion arena will use synthetic fallback."
        )
    else:
        try:
            download_celeba(data_root, force=force)
        except Exception as exc:
            LOGGER.error(
                "CelebA download failed (%s). Diffusion may fall back to synthetic data.", exc
            )

    status = dataset_status(data_root)
    manifest_path = os.path.join(data_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({"data_root": os.path.abspath(data_root), "ready": status}, fh, indent=2)
    LOGGER.info("Manifest written: %s", manifest_path)
    LOGGER.info("Dataset status: %s", status)
    return status


def print_upload_instructions(data_root: str) -> None:
    """Print how to copy the prepared folder to a cloud GPU."""
    root = os.path.abspath(data_root)
    LOGGER.info("--- Upload to RunPod / Jupyter ---")
    LOGGER.info("1. On your PC, archive the folder:")
    LOGGER.info(
        "   tar -czf fairbench_data.tar.gz -C %s .", os.path.dirname(root) if root != "/" else root
    )
    LOGGER.info("2. Upload fairbench_data.tar.gz to /workspace on the pod.")
    LOGGER.info("3. Extract:")
    LOGGER.info("   mkdir -p /workspace/data && tar -xzf fairbench_data.tar.gz -C /workspace/data")
    LOGGER.info("4. Run benchmark offline:")
    LOGGER.info(
        "   python -m fairbench --data-root /workspace/data --offline --output-dir results/full"
    )
