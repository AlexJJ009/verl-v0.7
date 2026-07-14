from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def records(repo_root: Path, requested: list[str]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for requested_path in sorted(requested):
        path = repo_root / requested_path
        if not path.exists() and not path.is_symlink():
            output.append({"path": requested_path, "type": "absent"})
            continue
        paths = [path]
        if path.is_dir() and not path.is_symlink():
            paths.extend(sorted(path.rglob("*"), key=lambda item: item.as_posix()))
        for item in paths:
            relative = item.relative_to(repo_root).as_posix()
            info = os.lstat(item)
            if item.is_symlink():
                output.append({"path": relative, "type": "symlink", "target": os.readlink(item)})
            elif item.is_dir():
                output.append({"path": relative, "type": "directory"})
            elif item.is_file():
                output.append({"path": relative, "type": "file", "bytes": info.st_size, "sha256": digest(item)})
            else:
                output.append({"path": relative, "type": "other", "mode": info.st_mode})
    return sorted(output, key=lambda item: str(item["path"]).encode())


def protected_roots(expected: list[dict[str, object]]) -> list[str]:
    roots: list[str] = []
    directory_roots: list[str] = []
    for item in sorted(expected, key=lambda value: (str(value["path"]).count("/"), str(value["path"]))):
        path = str(item["path"])
        if any(path.startswith(f"{root}/") for root in directory_roots):
            continue
        roots.append(path)
        if item.get("type") == "directory":
            directory_roots.append(path)
    return roots


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("capture", "compare"):
        command = sub.add_parser(action)
        command.add_argument("--repo-root", type=Path, required=True)
        command.add_argument("--path", action="append")
        command.add_argument("--baseline", type=Path)
        command.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "capture":
        if not args.path or not args.output:
            parser.error("capture requires --path and --output")
        value = records(args.repo_root.resolve(), args.path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in value))
        return 0
    if not args.baseline:
        parser.error("compare requires --baseline")
    expected = [json.loads(line) for line in args.baseline.read_text().splitlines() if line.strip()]
    requested = protected_roots(expected)
    actual = records(args.repo_root.resolve(), requested)
    if expected != actual:
        print(json.dumps({"ok": False, "expected": expected, "actual": actual}, sort_keys=True))
        return 1
    print(json.dumps({"ok": True, "sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
