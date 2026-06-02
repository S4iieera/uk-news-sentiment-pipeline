from __future__ import annotations
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dirs(data_dir: str | Path, run_id: str) -> dict[str, Path]:
    base = Path(data_dir)
    raw = ensure_dir(base / "raw" / run_id)
    processed = ensure_dir(base / "processed" / run_id)
    return {
        "raw": raw,
        "processed": processed,
    }
