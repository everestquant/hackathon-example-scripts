#!/usr/bin/env python3
"""Obfuscation gate for a public repository.

The term list is NOT part of this repository. It is supplied at runtime from
the OBFUSCATION_BLOCKLIST secret and lives only in the runner's memory for the
duration of the job.

Because this repository is public, so are its CI logs. This scanner therefore
reports only *coordinates* -- file, line, category, severity, and a stable
opaque id -- and never the matched term or the matched line. Resolve an id back
to a term with the canonical list in the private platform repository:

    awk -F, 'NR>1{print $1}' data/config/obfuscation_blocklist.csv \
      | while read -r t; do printf '%s %s\n' \
          "$(printf '%s' "$t" | shasum -a 256 | cut -c1-8)" "$t"; done

Usage:  OBFUSCATION_BLOCKLIST=<path-to-csv> python3 obfuscation_scan.py
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

CSV_COLUMNS_REQUIRED = {"term", "category", "severity"}
BINARY_SNIFF_BYTES = 4096


def load_terms(path: Path) -> list[tuple[re.Pattern[str], str, str, str]]:
    """Compile the blocklist. A `regex:` prefix means the term is already a pattern."""
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = CSV_COLUMNS_REQUIRED - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"blocklist is missing required column(s): {sorted(missing)}")

        compiled: list[tuple[re.Pattern[str], str, str, str]] = []
        for row in reader:
            raw = (row.get("term") or "").strip()
            if not raw:
                continue
            pattern = raw[6:] if raw.startswith("regex:") else r"\b" + re.escape(raw) + r"\b"
            try:
                rx = re.compile(pattern, re.IGNORECASE)
            except re.error:
                # A malformed row must not silently disable the whole gate.
                print(f"::warning::skipping unparsable blocklist row (id={term_id(raw)})")
                continue
            compiled.append(
                (rx, row.get("category", "?"), (row.get("severity") or "?").lower(), term_id(raw))
            )
    return compiled


def term_id(term: str) -> str:
    """Opaque, stable handle for a term. Safe to print in a public log.

    Not a security primitive -- it only has to be stable and non-reversing.
    SHA-256 rather than SHA-1 so the repo's own Semgrep gate stays green.
    """
    return hashlib.sha256(term.encode("utf-8")).hexdigest()[:8]


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, text=True, check=True
    ).stdout
    return [Path(p) for p in out.split("\0") if p]


def is_text(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\0" not in handle.read(BINARY_SNIFF_BYTES)
    except OSError:
        return False


def main() -> int:
    blocklist = os.environ.get("OBFUSCATION_BLOCKLIST", "")
    if not blocklist:
        sys.exit("OBFUSCATION_BLOCKLIST is unset -- refusing to report a false all-clear.")

    path = Path(blocklist)
    if not path.is_file():
        sys.exit(f"blocklist not found at {path} -- refusing to report a false all-clear.")

    terms = load_terms(path)
    if not terms:
        sys.exit("blocklist parsed to zero terms -- refusing to report a false all-clear.")

    violations: list[tuple[str, int, str, str, str]] = []
    scanned = 0
    for file_path in tracked_files():
        if not file_path.is_file() or not is_text(file_path):
            continue
        scanned += 1
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(content.splitlines(), 1):
            for rx, category, severity, tid in terms:
                if rx.search(line):
                    violations.append((str(file_path), lineno, category, severity, tid))

    print(f"Scanned {scanned} tracked text file(s) against {len(terms)} term(s).")

    if not violations:
        print("No blocklisted term found.")
        return 0

    critical = [v for v in violations if v[3] == "critical"]
    print(f"\n{len(violations)} match(es), {len(critical)} critical:\n")
    for file_path, lineno, category, severity, tid in violations:
        # Deliberately no term, no line content -- these logs are public.
        level = "error" if severity == "critical" else "warning"
        print(
            f"::{level} file={file_path},line={lineno}::"
            f"blocklisted term (category={category}, severity={severity}, id={tid})"
        )

    if critical:
        print(
            "\nA critical term reached a public file. Resolve the id against the "
            "canonical list in the private platform repo (see this script's docstring)."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
