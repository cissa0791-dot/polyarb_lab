from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_ENV_FILE_CANDIDATES = (
    Path("/root/.polymarket_env"),
    Path("/root/polyarb_lab/.env"),
)


def resolve_env_file(path: str | Path | None, *, root: str | Path | None = None) -> Path | None:
    """Resolve an explicit or safe default live env file path without reading secrets."""
    if path:
        return Path(path)

    env_override = os.environ.get("POLYARB_LIVE_ENV_FILE", "").strip()
    if env_override:
        return Path(env_override)

    candidates = list(DEFAULT_ENV_FILE_CANDIDATES)
    if root is not None:
        candidates.append(Path(root) / ".env")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_env_file(path: str | Path | None, *, override: bool = False) -> dict[str, Any]:
    """Load KEY=VALUE pairs from an env file without exposing values."""
    if path is None:
        return {"loaded": False, "reason": "NO_ENV_FILE", "keys": []}
    source = Path(path)
    if not source.exists():
        return {"loaded": False, "reason": "ENV_FILE_NOT_FOUND", "path": str(source), "keys": []}

    keys: list[str] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = _strip_inline_comment(value.strip())
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value
        keys.append(key)

    return {"loaded": True, "path": str(source), "keys": sorted(set(keys))}


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if index == 0 or value[index - 1].isspace():
                return value[:index].strip()
    return value
