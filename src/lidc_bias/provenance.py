"""Provenance helpers for reproducible and auditable analysis runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "krippendorff",
    "matplotlib",
    "numpy",
    "pandas",
    "pyarrow",
    "pydicom",
    "scikit-learn",
    "scipy",
    "xlrd",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_identifier(*parts: str, length: int = 24) -> str:
    """Create a stable non-clinical identifier from canonical source fields."""
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_revision(project_root: Path) -> str | None:
    """Return the current Git revision when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def base_manifest(project_root: Path, config_path: Path, stage: str) -> dict[str, Any]:
    """Return common run metadata for a pipeline-stage manifest."""
    return {
        "stage": stage,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(project_root),
        "config_path": str(config_path.relative_to(project_root)),
        "config_sha256": sha256_file(config_path),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write formatted JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
