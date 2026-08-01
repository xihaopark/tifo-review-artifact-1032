"""Write inspectable provenance beside every generated metric file."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def write_run_manifest(args: Any, setting: str, output_dir: str, metrics: dict[str, float]) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.root_path) / args.data_path
    dirty = _git_value(["status", "--porcelain"])
    payload = {
        "schema_version": "tifo-run-v1",
        "setting": setting,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "method": getattr(args, "method", "tifo"),
        "seed": args.random_seed,
        "task": {
            "dataset": args.data_path,
            "data_type": args.data,
            "data_path": str(data_path.resolve()),
            "data_sha256": _sha256(data_path),
            "seq_len": args.seq_len,
            "label_len": args.label_len,
            "pred_len": args.pred_len,
            "features": args.features,
        },
        "provenance": {
            "git_revision": _git_value(["rev-parse", "HEAD"]),
            "git_dirty": bool(dirty),
            "git_status": dirty.splitlines() if dirty else [],
        },
        "arguments": {key: _json_safe(value) for key, value in vars(args).items()},
        "metrics": metrics,
        "artifacts": {
            "metrics": str((output / "metrics.npy").resolve()),
            **(
                {
                    "predictions": str((output / "pred.npy").resolve()),
                    "targets": str((output / "true.npy").resolve()),
                }
                if getattr(args, "save_arrays", False)
                else {}
            ),
        },
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path
