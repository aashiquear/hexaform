#!/usr/bin/env python3
"""Render the version-controlled technical report as Markdown and HTML."""

from __future__ import annotations

import argparse
import html
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def git_value(*args: str, default: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "unknown"


def markdown_to_html(markdown: str, title: str) -> str:
    """Produce a dependency-free, printable HTML representation."""
    escaped = html.escape(markdown)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{max-width:980px;margin:3rem auto;padding:0 2rem;color:#172033;
font:16px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
@media print{{body{{margin:0;max-width:none}}}}
</style></head><body><pre>{escaped}</pre></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "reports")
    args = parser.parse_args()

    version = project_version()
    commit = git_value("rev-parse", "--short=12", "HEAD", default="uncommitted")
    dirty = git_value("status", "--porcelain", default="")
    if dirty:
        commit += "+dirty"
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    source = (ROOT / "docs" / "TECHNICAL_REPORT.md").read_text(encoding="utf-8")
    rendered = (source.replace("{{VERSION}}", version)
                .replace("{{COMMIT}}", commit)
                .replace("{{GENERATED_AT}}", generated))
    safe_commit = commit.replace("+", "-")
    stem = f"hexaform-technical-report-v{version}-{safe_commit}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / f"{stem}.md"
    html_path = args.output_dir / f"{stem}.html"
    md_path.write_text(rendered, encoding="utf-8")
    html_path.write_text(markdown_to_html(rendered, f"Hexaform {version} technical report"), encoding="utf-8")
    print(md_path)
    print(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
