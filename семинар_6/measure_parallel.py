
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orchestrator import (
    _topological_levels,
    execute_level,
    execute_level_sequential,
)
from planner import planner
from schema_validator import validate_plan

OUT = Path(__file__).resolve().parent / "parallel.json"

QUERIES = {
    "Q1": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
    "Q5": "Каковы были курс доллара, ключевая ставка ЦБ и инфляция (ИПЦ г/г) на 15 марта 2026?",
}


def _build_plan(query: str):
    plan = planner(query)
    errs = validate_plan(plan)
    if errs:
        plan = planner(query, feedback=f"Инструменты не существуют: {errs}")
    return plan


def _time(levels, runner) -> float:
    answers = {}
    t0 = time.perf_counter()
    for level in levels:
        answers.update(runner(level, answers))
    return time.perf_counter() - t0


def measure(qid: str, query: str) -> dict:
    plan = _build_plan(query)
    levels = _topological_levels(plan.subquestions)
    sizes = [len(l) for l in levels]
    print(f"\n[{qid}] подвопросов={len(plan.subquestions)}, уровни(размеры)={sizes}")

    seq = _time(levels, execute_level_sequential)
    print(f"  последовательно: {seq:.2f} c")
    par = _time(levels, execute_level)
    print(f"  параллельно:     {par:.2f} c")
    speedup = seq / par if par > 0 else float("nan")
    print(f"  ускорение: x{speedup:.2f}")
    return {
        "qid": qid, "query": query, "n_subquestions": len(plan.subquestions),
        "level_sizes": sizes, "sequential_sec": round(seq, 3),
        "parallel_sec": round(par, 3), "speedup": round(speedup, 3),
    }


def run() -> dict:
    rows = [measure(qid, q) for qid, q in QUERIES.items()]
    payload = {"rows": rows}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {OUT}")
    return payload


if __name__ == "__main__":
    print("Замер ускорения (последовательно vs параллельно)...")
    run()
