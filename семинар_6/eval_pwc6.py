

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_s5 import run_agent
from eval_pwc import CASES as BASE_CASES, _check_pwc, _check_single
from orchestrator import run_pwc

EXTRA_CASES = [
    {
        "id": "Q4",
        "query": "Каков среднегодовой темп роста (CAGR) курса доллара США с января 2022 по январь 2026?",
        "comment": (
            "validator-fixable: Планировщик склонен выдумать get_cagr/get_average. "
            "Одиночный и PWC без валидатора дают галлюцинацию инструмента; "
            "PWC + валидатор переразбивает на get_fx_rate + calculate."
        ),
        "expected_tools_pwc": {"get_fx_rate", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q5",
        "query": "Каковы были курс доллара, ключевая ставка ЦБ и инфляция (ИПЦ г/г) на 15 марта 2026?",
        "comment": (
            "Естественная параллельность: 3 независимых подвопроса "
            "(get_fx_rate, get_key_rate, get_inflation), depends_on пуст. "
            "На нём же меряем ускорение из части 2."
        ),
        "expected_tools_pwc": {"get_fx_rate", "get_key_rate", "get_inflation"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q6",
        "query": (
            "Что в декабре 2024 было выше — ключевая ставка ЦБ или годовая инфляция "
            "(ИПЦ г/г), и на сколько процентных пунктов они отличались?"
        ),
        "comment": "Личный макро-вопрос: реальная ставка = ставка − инфляция г/г.",
        "expected_tools_pwc": {"get_key_rate", "get_inflation", "calculate"},
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
]

CASES = BASE_CASES + EXTRA_CASES
CONFIGS = ["single", "pwc_noval", "pwc_val"]


def _run(config: str, query: str):
    case = _case_by_query(query)
    if config == "single":
        r = run_agent(query, max_iter=8, verbose=False)
        return _check_single(case, r)
    use_val = config == "pwc_val"
    r = run_pwc(query, max_iter=3, verbose=False, use_validator=use_val)
    return _check_pwc(case, r)


def _case_by_query(query: str) -> dict:
    for c in CASES:
        if c["query"] == query:
            return c
    return {"id": "_", "must_have_keywords": [], "forbid_hallucinated_tools": True}


def run(n: int) -> dict:
    results = {}
    for case in CASES:
        qid = case["id"]
        results[qid] = {"query": case["query"], "configs": {}}
        for cfg in CONFIGS:
            passes = 0
            for i in range(n):
                try:
                    chk = _run(cfg, case["query"])
                    ok = bool(chk["ok"])
                    prev = chk.get("answer_preview", "")
                except Exception as exc:  
                    ok, prev = False, f"EXC: {exc}"
                passes += int(ok)
                print(f"  {qid} [{cfg}] {i+1}/{n}: {'PASS' if ok else 'fail'} :: {prev[:70]}")
            results[qid]["configs"][cfg] = {"passes": passes, "N": n}
            print(f"  => {qid} [{cfg}]: {passes}/{n}")

    payload = {"N": n, "configs": CONFIGS, "results": results}
    out = Path(__file__).resolve().parent / "eval6.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n| id | single | pwc | pwc+вал |")
    print("|---|---|---|---|")
    for qid, r in results.items():
        c = r["configs"]
        print(f"| {qid} | {c['single']['passes']}/{n} | {c['pwc_noval']['passes']}/{n} | {c['pwc_val']['passes']}/{n} |")
    print(f"\nСохранено: {out}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true", help="N=1, быстрая проверка")
    ap.add_argument("-n", type=int, default=int(os.environ.get("EVAL_N", "5")))
    args = ap.parse_args()
    n = 1 if args.single else args.n
    print(f"Eval С6: {len(CASES)} вопросов × {len(CONFIGS)} конфигурации × N={n} (~{len(CASES)*len(CONFIGS)*n} запусков)\n")
    run(n)
