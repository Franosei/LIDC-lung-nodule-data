"""Bind completed analysis outputs to a clean, fixed Git source snapshot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lidc_bias.provenance import analysis_code_sha256, sha256_file, write_json  # noqa: E402

MANIFEST_PATHS = (
    Path("data/processed/ingestion_manifest.json"),
    Path("results/label_case_effects/manifest.json"),
    Path("results/shared_assessment/manifest.json"),
    Path("results/publication/manifest.json"),
)


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_files(config_path: Path) -> list[Path]:
    files = sorted((PROJECT_ROOT / "src" / "lidc_bias").rglob("*.py"))
    files.extend([PROJECT_ROOT / "pyproject.toml", config_path])
    return sorted(set(files))


def _source_inventory(config_path: Path) -> list[dict[str, str]]:
    return [
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in _source_files(config_path)
    ]


def finalize(tag: str) -> Path:
    """Verify all recorded hashes and write release bindings without refitting."""
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("The worktree must be clean before finalizing a release")
    revision = _git("rev-parse", "HEAD")
    config_path = PROJECT_ROOT / "config" / "analysis.toml"
    current_code_hash = analysis_code_sha256(PROJECT_ROOT, config_path)
    current_config_hash = sha256_file(config_path)
    verified_outputs: list[dict[str, str]] = []
    updated_manifests: list[tuple[Path, dict]] = []

    for relative_manifest in MANIFEST_PATHS:
        manifest_path = PROJECT_ROOT / relative_manifest
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("analysis_code_sha256") != current_code_hash:
            raise RuntimeError(f"Code hash mismatch in {relative_manifest}")
        if manifest.get("config_sha256") != current_config_hash:
            raise RuntimeError(f"Configuration hash mismatch in {relative_manifest}")
        for output_name, output in manifest["outputs"].items():
            output_path = PROJECT_ROOT / output["path"]
            observed = sha256_file(output_path)
            if observed != output["sha256"]:
                raise RuntimeError(
                    f"Output hash mismatch: {manifest['stage']}:{output_name}"
                )
            verified_outputs.append(
                {
                    "stage": manifest["stage"],
                    "output": output_name,
                    "path": output_path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": observed,
                }
            )
        manifest["release_source"] = {
            "git_revision": revision,
            "git_tag": tag,
            "git_worktree_dirty": False,
            "analysis_code_sha256": current_code_hash,
            "config_sha256": current_config_hash,
            "source_snapshot_verified": True,
            "output_hashes_verified": True,
        }
        updated_manifests.append((manifest_path, manifest))

    for manifest_path, manifest in updated_manifests:
        write_json(manifest_path, manifest)

    release_dir = PROJECT_ROOT / "release"
    record_path = release_dir / "analysis_source_release.json"
    manifest_hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
        for path, _ in updated_manifests
    }
    record = {
        "release_tag": tag,
        "analysis_source_git_revision": revision,
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "analysis_code_sha256": current_code_hash,
        "config_sha256": current_config_hash,
        "historical_execution_note": (
            "The original run manifests correctly recorded a dirty worktree. "
            "The release_source block binds the unchanged executed analysis code "
            "to this clean Git snapshot; models were not refitted."
        ),
        "source_files": _source_inventory(config_path),
        "verified_outputs": verified_outputs,
        "release_manifest_sha256": manifest_hashes,
    }
    write_json(record_path, record)
    return record_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Fixed Git tag for the analysis release")
    arguments = parser.parse_args()
    record = finalize(arguments.tag)
    print(record.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
