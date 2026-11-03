"""Verify the fixed analysis source snapshot, manifests and result files."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = PROJECT_ROOT / "release" / "analysis_source_release.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    revision = record["analysis_source_git_revision"]
    failures: list[str] = []

    for source in record["source_files"]:
        relative = source["path"]
        expected = source["sha256"]
        if _sha256(PROJECT_ROOT / relative) != expected:
            failures.append(f"current source mismatch: {relative}")
        blob = subprocess.run(
            ["git", "show", f"{revision}:{relative}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() != expected:
            failures.append(f"Git source mismatch: {relative}")

    for output in record["verified_outputs"]:
        if _sha256(PROJECT_ROOT / output["path"]) != output["sha256"]:
            failures.append(f"result mismatch: {output['path']}")

    for relative, expected in record["release_manifest_sha256"].items():
        path = PROJECT_ROOT / relative
        if _sha256(path) != expected:
            failures.append(f"manifest mismatch: {relative}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        release = manifest.get("release_source", {})
        if release.get("git_revision") != revision:
            failures.append(f"release revision mismatch: {relative}")
        if release.get("git_worktree_dirty") is not False:
            failures.append(f"release is not marked clean: {relative}")
        if release.get("analysis_code_sha256") != record["analysis_code_sha256"]:
            failures.append(f"code hash mismatch: {relative}")
        if release.get("output_hashes_verified") is not True:
            failures.append(f"outputs not verified: {relative}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(
        f"Verified {len(record['source_files'])} source files, "
        f"{len(record['verified_outputs'])} outputs and "
        f"{len(record['release_manifest_sha256'])} manifests "
        f"against {revision}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
