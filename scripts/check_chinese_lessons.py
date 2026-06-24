#!/usr/bin/env python3
"""QA checks for Chinese localized lesson Markdown.

Usage:
    python3 scripts/check_chinese_lessons.py [--phase N] [--json]

Exit codes:
    0 — clean
    1 — issues found
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
PHASES_DIR = ROOT / "phases"

PHASE_DIR_RE = re.compile(r"^[0-9]{2}-[a-z0-9][a-z0-9-]*[a-z0-9]$")
CODE_FENCE_RE = re.compile(r"^```([^\s`]*)[^\n]*$")
INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$",
    re.MULTILINE,
)

REQUIRED_SECTION_PREFIXES: tuple[str, ...] = (
    "Learning Objectives",
    "Build It",
    "Use It",
    "Ship It",
    "Exercises",
    "Key Terms",
)

MIN_INLINE_CODE_RATIO = 0.8


@dataclass(frozen=True)
class Issue:
    rule: str
    file: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "file": self.file, "message": self.message}


@dataclass
class Report:
    zh_files_checked: int = 0
    issues: list[Issue] = field(default_factory=list)

    def add(self, rule: str, file: Path, message: str) -> None:
        self.issues.append(
            Issue(rule, file.relative_to(ROOT).as_posix(), message)
        )


def iter_lesson_dirs(phase_filter: int | None) -> Iterable[Path]:
    if not PHASES_DIR.is_dir():
        return
    for phase in sorted(PHASES_DIR.iterdir()):
        if not phase.is_dir() or not PHASE_DIR_RE.match(phase.name):
            continue
        if phase_filter is not None:
            try:
                phase_num = int(phase.name.split("-", 1)[0])
            except ValueError:
                continue
            if phase_num != phase_filter:
                continue
        for lesson in sorted(phase.iterdir()):
            if lesson.is_dir():
                yield lesson


def read_text(report: Report, path: Path, rule: str) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        report.add(rule, path, "file is not valid UTF-8")
    except OSError as exc:
        report.add(rule, path, f"could not read file: {exc}")
    return None


def fenced_blocks(text: str) -> list[tuple[str, tuple[str, ...]]]:
    blocks: list[tuple[str, tuple[str, ...]]] = []
    in_block = False
    language = ""
    body: list[str] = []
    for line in text.splitlines():
        if not in_block:
            match = CODE_FENCE_RE.match(line)
            if match:
                in_block = True
                language = match.group(1)
                body = []
            continue
        if line.strip() == "```":
            blocks.append((language, tuple(body)))
            in_block = False
            language = ""
            body = []
        else:
            body.append(line)
    return blocks


def normalize_fenced_blocks(
    blocks: list[tuple[str, tuple[str, ...]]],
) -> list[tuple[str, tuple[str, ...]]]:
    return [
        (language, tuple(line.rstrip() for line in body))
        for language, body in blocks
    ]


def markdown_headings(text: str) -> set[str]:
    headings: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        marker, _, title = line.partition(" ")
        if marker and set(marker) == {"#"} and title:
            headings.add(title.strip())
    return headings


def missing_required_sections(text: str) -> list[str]:
    headings = markdown_headings(text)
    missing: list[str] = []
    for prefix in REQUIRED_SECTION_PREFIXES:
        if not any(heading.startswith(prefix) for heading in headings):
            missing.append(prefix)
    return missing


def check_pair(report: Report, en_path: Path, zh_path: Path) -> None:
    report.zh_files_checked += 1

    en_text = read_text(report, en_path, "ZH001")
    zh_text = read_text(report, zh_path, "ZH001")
    if en_text is None or zh_text is None:
        return

    en_fences = fenced_blocks(en_text)
    zh_fences = fenced_blocks(zh_text)
    en_languages = [language for language, _body in en_fences]
    zh_languages = [language for language, _body in zh_fences]
    if zh_languages != en_languages:
        report.add(
            "ZH002",
            zh_path,
            "fenced code block languages differ from docs/en.md "
            f"(en={en_languages!r}, zh={zh_languages!r})",
        )
    elif normalize_fenced_blocks(zh_fences) != normalize_fenced_blocks(en_fences):
        report.add(
            "ZH007",
            zh_path,
            "fenced code block contents differ from docs/en.md "
            "(ignoring trailing whitespace)",
        )

    en_inline = len(INLINE_CODE_RE.findall(en_text))
    zh_inline = len(INLINE_CODE_RE.findall(zh_text))
    min_expected = int(en_inline * MIN_INLINE_CODE_RATIO)
    if zh_inline < min_expected:
        report.add(
            "ZH003",
            zh_path,
            "inline code count is unexpectedly low compared with docs/en.md "
            f"(en={en_inline}, zh={zh_inline}, minimum={min_expected})",
        )

    missing = missing_required_sections(zh_text)
    if missing:
        report.add(
            "ZH004",
            zh_path,
            "missing required section heading prefix(es): " + ", ".join(missing),
        )

    en_tables = len(TABLE_SEPARATOR_RE.findall(en_text))
    zh_tables = len(TABLE_SEPARATOR_RE.findall(zh_text))
    if zh_tables < en_tables:
        report.add(
            "ZH005",
            zh_path,
            "Markdown table separator lines decreased compared with docs/en.md "
            f"(en={en_tables}, zh={zh_tables})",
        )

    if not CJK_RE.search(zh_text):
        report.add("ZH006", zh_path, "localized file contains no CJK characters")


def run_checks(phase_filter: int | None) -> Report:
    report = Report()
    for lesson in iter_lesson_dirs(phase_filter):
        zh_path = lesson / "docs" / "zh.md"
        if not zh_path.exists():
            continue
        en_path = lesson / "docs" / "en.md"
        if not en_path.is_file():
            report.zh_files_checked += 1
            report.add(
                "ZH001",
                zh_path,
                "docs/zh.md exists but matching docs/en.md is missing",
            )
            continue
        check_pair(report, en_path, zh_path)
    return report


def render_text(report: Report) -> str:
    lines = [
        "check_chinese_lessons.py — "
        f"{report.zh_files_checked} zh lesson file(s) checked, "
        f"{len(report.issues)} issue(s)"
    ]
    if report.issues:
        lines.append("")
        for issue in report.issues:
            lines.append(f"  [{issue.rule}] {issue.file}: {issue.message}")
        lines.append("")
        by_rule: dict[str, int] = {}
        for issue in report.issues:
            by_rule[issue.rule] = by_rule.get(issue.rule, 0) + 1
        lines.append("Summary by rule:")
        for rule in sorted(by_rule):
            lines.append(f"  {rule}: {by_rule[rule]}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, help="only check one numeric phase")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    report = run_checks(args.phase)
    if args.json:
        payload = {
            "ok": not report.issues,
            "zh_files_checked": report.zh_files_checked,
            "issues": [issue.to_dict() for issue in report.issues],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(render_text(report), end="")
    return 1 if report.issues else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
