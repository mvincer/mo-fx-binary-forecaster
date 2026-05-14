"""Point cache/data loading at the sibling **Currencies** project (shared FXCM/FRED parquet)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

THIS_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv_for_paths() -> None:
    """So ``FXNL_CURRENCIES_ROOT`` in ``.env`` works for CLI and Streamlit."""
    try:
        from dotenv import load_dotenv

        load_dotenv(THIS_ROOT / ".env")
        sib = THIS_ROOT.parent / "Currencies" / ".env"
        nested = THIS_ROOT / "Currencies" / ".env"
        if nested.is_file():
            load_dotenv(nested)
        if sib.is_file():
            load_dotenv(sib)
    except Exception:
        pass


def currencies_root() -> Path:
    """
    ``.../Currencies`` next to this project folder, or **override**:

    Set environment variable **``FXNL_CURRENCIES_ROOT``** to an absolute path to the
    Currencies repo whose ``data/cache/fxcm/*.parquet`` you keep updated (useful if you
    have several clones, OneDrive copies, or the sibling folder is not the one you refresh).
    """
    _load_dotenv_for_paths()
    env_raw = (os.environ.get("FXNL_CURRENCIES_ROOT") or "").strip()
    if env_raw:
        env_p = Path(env_raw).expanduser().resolve()
        if env_p.is_dir() and (env_p / "src" / "cache_paths.py").is_file():
            return env_p
        raise FileNotFoundError(
            "FXNL_CURRENCIES_ROOT is set but that folder is missing or incomplete.\n"
            f"  Path: {env_p}\n"
            "  Expected: src/cache_paths.py (a full Currencies / Simple Regression Based (FX) checkout)."
        )

    cand = THIS_ROOT.parent / "Currencies"
    if cand.is_dir() and (cand / "src" / "cache_paths.py").is_file():
        return cand.resolve()
    nested = THIS_ROOT / "Currencies"
    if nested.is_dir() and (nested / "src" / "cache_paths.py").is_file():
        return nested.resolve()
    raise FileNotFoundError(
        f"Sibling Currencies project not found or incomplete. Expected:\n"
        f"  {cand}\nwith src/cache_paths.py (clone or copy **Simple Regression Based (FX)** next to this folder)."
    )


def patch_currencies_sys_path() -> Path:
    """
    Prepend Currencies to ``sys.path`` so ``import src.merged_data`` resolves to **that** repo,
    then refresh ``src.cache_paths`` roots.

    This app uses package **fxnl** (not ``src``) so it does not shadow Currencies' ``src``.
    """
    root = currencies_root()
    rs = str(root)
    if rs in sys.path:
        sys.path.remove(rs)
    sys.path.insert(0, rs)

    import src.cache_paths as cp  # noqa: PLC0415 — Currencies project

    cp.ROOT = root
    cp.DATA_CACHE = root / "data" / "cache"
    cp.FXCM_DIR = cp.DATA_CACHE / "fxcm"
    cp.FRED_DIR = cp.DATA_CACHE / "fred"
    cp.MANIFEST_PATH = cp.DATA_CACHE / "manifest.json"
    return root


def fxcm_primary_parquet_path(primary: str) -> Path:
    """
    Absolute path to the FXCM daily parquet for ``primary`` (e.g. ``EUR/USD``),
    same rule as :func:`src.fxcm_loader.instrument_cache_meta`.
    """
    patch_currencies_sys_path()
    from src.cache_paths import FXCM_DIR  # noqa: PLC0415

    sym = re.sub(r"[^\w]+", "_", primary.strip()).strip("_")
    return (FXCM_DIR / f"{sym}.parquet").resolve()


def fxcm_primary_parquet_mtime(primary: str) -> float:
    """File ``st_mtime`` for cache invalidation (changes after Currencies refresh)."""
    p = fxcm_primary_parquet_path(primary)
    try:
        return float(p.stat().st_mtime)
    except OSError:
        return -1.0
