"""Shared experiment constants and serialization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CLASSES = ("cat", "fish", "airplane")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/phase0.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
