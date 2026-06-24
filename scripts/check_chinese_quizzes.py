#!/usr/bin/env python3
"""QA checks for localized quiz.zh.json files."""

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
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
REQUIRED_KEYS = {"stage", "question", "options", "correct", "explanation"}


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

    def add(self, rule: str, path: Path, message: str) -> None:
        self.issues.append(Issue(rule, path.relative_to(ROOT).as_posix(), message))


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


def load_json(report: Report, path: Path, rule: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.add(rule, path, f"invalid JSON: {exc}")
    except UnicodeDecodeError:
        report.add(rule, path, "file is not valid UTF-8")
    except OSError as exc:
        report.add(rule, path, f"could not read file: {exc}")
    return None


def questions(data) -> list[dict]:
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        return data["questions"]
    if isinstance(data, list):
        return data
    return []


def check_pair(report: Report, en_path: Path, zh_path: Path) -> None:
    report.zh_files_checked += 1
    en_data = load_json(report, en_path, "ZQ001")
    zh_data = load_json(report, zh_path, "ZQ001")
    if en_data is None or zh_data is None:
        return

    en_qs = questions(en_data)
    zh_qs = questions(zh_data)
    if len(en_qs) != len(zh_qs):
        report.add("ZQ002", zh_path, f"question count differs from quiz.json (en={len(en_qs)}, zh={len(zh_qs)})")
        return

    for idx, (en_q, zh_q) in enumerate(zip(en_qs, zh_qs), start=1):
        missing = REQUIRED_KEYS - set(zh_q)
        if missing:
            report.add("ZQ003", zh_path, f"question {idx} missing keys: {', '.join(sorted(missing))}")
            continue
        if zh_q.get("stage") != en_q.get("stage"):
            report.add("ZQ004", zh_path, f"question {idx} stage changed from {en_q.get('stage')!r} to {zh_q.get('stage')!r}")
        if zh_q.get("correct") != en_q.get("correct"):
            report.add("ZQ005", zh_path, f"question {idx} correct index changed from {en_q.get('correct')!r} to {zh_q.get('correct')!r}")
        if len(zh_q.get("options") or []) != len(en_q.get("options") or []):
            report.add("ZQ006", zh_path, f"question {idx} option count differs from quiz.json")

    zh_text = zh_path.read_text(encoding="utf-8", errors="ignore")
    if not CJK_RE.search(zh_text):
        report.add("ZQ007", zh_path, "localized quiz contains no CJK characters")


def run_checks(phase_filter: int | None, require_all: bool = False) -> Report:
    report = Report()
    for lesson in iter_lesson_dirs(phase_filter):
        zh_path = lesson / "quiz.zh.json"
        if not zh_path.exists():
            if require_all and (lesson / "quiz.json").is_file():
                report.add("ZQ008", zh_path, "matching quiz.json exists but quiz.zh.json is missing")
            continue
        en_path = lesson / "quiz.json"
        if not en_path.is_file():
            report.zh_files_checked += 1
            report.add("ZQ001", zh_path, "quiz.zh.json exists but matching quiz.json is missing")
            continue
        check_pair(report, en_path, zh_path)
    return report


def render_text(report: Report) -> str:
    lines = [
        "check_chinese_quizzes.py — "
        f"{report.zh_files_checked} zh quiz file(s) checked, "
        f"{len(report.issues)} issue(s)"
    ]
    if report.issues:
        lines.append("")
        for issue in report.issues:
            lines.append(f"  [{issue.rule}] {issue.file}: {issue.message}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, help="only check one numeric phase")
    parser.add_argument("--require-all", action="store_true", help="require every quiz.json to have quiz.zh.json")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    report = run_checks(args.phase, args.require_all)
    if args.json:
      print(json.dumps({
          "ok": not report.issues,
          "zh_files_checked": report.zh_files_checked,
          "issues": [issue.to_dict() for issue in report.issues],
      }, indent=2, ensure_ascii=False))
    else:
      print(render_text(report), end="")
    return 1 if report.issues else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
