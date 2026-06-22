
#Подготовка eval-набора: стратифицированная выборка пар из input/test.csv (реальный датасет cnamuangtoun/resume-job-description-fit, метки No Fit / Potential Fit / Good Fit).


from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEST_CSV = HERE / "input" / "test.csv"
TRAIN_CSV = HERE / "input" / "train.csv"
OUT = HERE / "input" / "eval_set.jsonl"
LABELS = ("No Fit", "Potential Fit", "Good Fit")


def load_csv(path: Path) -> list[dict]:
    csv.field_size_limit(10_000_000)
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


import re as _re

_RESUME_SKILL_MARK = _re.compile(r"\bskill|\bproficien|\btechnolog|\bcompetenc", _re.I)
_JD_SKILL_MARK = _re.compile(r"\bskill|\brequire|\bqualif|\bproficien|\bmust have", _re.I)


def _has_explicit_skills(r: dict) -> bool:
    return bool(_RESUME_SKILL_MARK.search(r.get("resume_text", "")) and
                _JD_SKILL_MARK.search(r.get("job_description_text", "")))


def build(per_class: int, seed: int, source: Path, binary: bool = False,
          explicit_skills: bool = False) -> list[dict]:
    rows = load_csv(source)
    if explicit_skills:
        rows = [r for r in rows if _has_explicit_skills(r)]
    if binary:
        labels = ("No Fit", "Fit")
        by_label = {l: [] for l in labels}
        for r in rows:
            lab = r.get("label", "").strip()
            if lab not in LABELS:
                continue
            by_label["No Fit" if lab == "No Fit" else "Fit"].append(r)
    else:
        labels = LABELS
        by_label = {l: [] for l in labels}
        for r in rows:
            lab = r.get("label", "").strip()
            if lab in by_label:
                by_label[lab].append(r)

    rng = random.Random(seed)
    out: list[dict] = []
    for lab in labels:
        pool = by_label[lab]
        rng.shuffle(pool)
        for i, r in enumerate(pool[:per_class]):
            out.append({
                "id": f"{lab.replace(' ', '')}_{i+1}",
                "resume_text": r["resume_text"],
                "job_description_text": r["job_description_text"],
                "label": lab,  # в бинарном режиме здесь уже 'No Fit'/'Fit'
            })
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=6, help="пар на каждый класс")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source", choices=["test", "train"], default="test")
    ap.add_argument("--binary", action="store_true",
                    help="бинарный режим: No Fit vs Fit (Potential+Good вместе)")
    ap.add_argument("--explicit-skills", action="store_true",
                    help="только пары, где навыки явно перечислены и в резюме, и в вакансии")
    args = ap.parse_args()

    src = TEST_CSV if args.source == "test" else TRAIN_CSV
    if not src.exists():
        sys.exit(f"Нет файла {src}. Положите train.csv/test.csv в input/ (см. README).")

    per_class = args.per_class
    if args.binary and per_class < 8:
        per_class = 12  #

    data = build(per_class, args.seed, src, binary=args.binary,
                 explicit_skills=args.explicit_skills)
    with open(OUT, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    dist = {}
    for d in data:
        dist[d["label"]] = dist.get(d["label"], 0) + 1
    print(f"Сохранено {len(data)} пар в {OUT} ({'binary' if args.binary else '3-class'})")
    print("Распределение меток:", dist)


if __name__ == "__main__":
    main()
