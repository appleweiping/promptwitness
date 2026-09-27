"""Check every sealed historical member without extracting or altering history."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def check(root: Path, archive: Path) -> dict:
    expected = "8674f1c21f2e223f96e73301b2050a702d722f5040f4ba9b01b062a389bd01aa"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError("historical archive changed")
    files = {}
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            if member.is_dir():
                continue
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in member.filename:
                raise ValueError("invalid sealed member identity")
            path = root.joinpath(*name.parts)
            if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError("missing or external historical file")
            raw = zipped.read(member)
            if path.read_bytes() != raw:
                raise ValueError(f"historical file changed: {name}")
            files[str(name)] = hashlib.sha256(raw).hexdigest()
    return {
        "format": "promptwitness.delta.history-seal-audit/v1.1",
        "archive_sha256": expected,
        "historical_members_byte_identical": True,
        "member_count": len(files),
        "source_sha256": files,
        "history_overwritten": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = check(args.root, args.archive)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("archive_sha256", "historical_members_byte_identical", "member_count")
            }
        )
    )
