"""Verify a downloaded snapshot against metadata fetched from the official Hub."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(manifest_path: Path, snapshot: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    checked = []
    for entry in manifest["siblings"]:
        relative = PurePosixPath(entry["rfilename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe manifest path")
        path = snapshot.joinpath(*relative.parts)
        if not path.is_file() or path.stat().st_size != entry["size"]:
            failures.append({"file": str(relative), "reason": "missing_or_size_mismatch"})
            continue
        large = entry.get("lfs")
        digest = hashlib.sha256() if large else hashlib.sha1(usedforsecurity=False)
        if not large:
            digest.update(f"blob {entry['size']}\0".encode("ascii"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        expected = large["sha256"] if large else entry["blobId"]
        actual = digest.hexdigest()
        checked.append({"file": str(relative), "digest": actual})
        if actual != expected:
            failures.append({"file": str(relative), "reason": "hash_mismatch"})
    return {
        "format": "promptwitness.delta.artifact-verification/v1",
        "model": manifest["id"],
        "revision": manifest["sha"],
        "passed": not failures,
        "checked": checked,
        "failures": failures,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    result = verify(args.manifest, args.snapshot)
    print(json.dumps(result, sort_keys=True, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
