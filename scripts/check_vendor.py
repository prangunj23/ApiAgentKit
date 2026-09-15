#!/usr/bin/env python3
"""Check this repo's vendored agentkit against the manifest recorded when it was synced.

agentkit is vendored here from the ApiAgentKit repo, which this repo does not check out. So
this catches the failure that is detectable without upstream: someone editing the vendored
copy in place. Such an edit is lost the next time ApiAgentKit syncs, so make it upstream.

    uv run chat_agent/vendor/check_vendor.py

Copied from ApiAgentKit/scripts/check_vendor.py by sync_vendor.py. Do not edit here.
"""

import hashlib
import json
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "agentkit"


def main() -> int:
    manifest_path = VENDOR / "VENDOR.json"
    if not manifest_path.is_file():
        print(f"No manifest at {manifest_path}", file=sys.stderr)
        return 1

    expected = json.loads(manifest_path.read_text())["files"]
    package = VENDOR / "src" / "agentkit"
    problems = []

    for relative, want in sorted(expected.items()):
        path = package / relative
        if not path.is_file():
            problems.append(f"missing: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != want:
            problems.append(f"edited in place: {relative}")

    found = {
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    problems.extend(f"not in manifest: {extra}" for extra in sorted(found - set(expected)))

    if problems:
        for problem in problems:
            print(problem)
        print(
            f"\n{len(problems)} problem(s). Vendored agentkit is edited or incomplete. "
            "Make the change in the ApiAgentKit repo and re-run its scripts/sync_vendor.py.",
            file=sys.stderr,
        )
        return 1

    print(f"vendored agentkit matches its manifest ({len(expected)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
