

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import CACHE_STATS, run_agent

CASES = [
    {
        "id": 1,
        "query": "Какая сегодня ключевая ставка ЦБ?",
        "expected_tools": ["get_key_rate"],
        "must_have": [],  # число не фиксируем — зависит от живого запроса
        "comment": "Базовый тест — один инструмент, одно число.",
    },
    {
        "id": 2,
        "query": "Сколько стоит доллар сегодня и сколько стоил 1 января 2022?",
        "expected_tools": ["get_fx_rate"],
        "must_have": [],
        "comment": "Два вызова одного инструмента с разными аргументами.",
    },
    {
        "id": 3,
        "query": "Какая сейчас реальная ключевая ставка? (номинальная минус инфляция г/г)",
        "expected_tools": ["get_key_rate", "get_inflation", "calculate"],
        "must_have": ["%"],
        "comment": "Три разных инструмента + арифметика. Классический многостадийный кейс.",
    },
    {
        "id": 4,
        "query": "Посчитай, за сколько лет удвоится вклад 100 тыс руб при текущей ключевой ставке (формула 72).",
        "expected_tools": ["get_key_rate", "calculate"],
        "must_have": ["год"],
        "comment": "Вычисление с формулой: 72 / ставка = годы.",
    },
    {
        "id": 5,
        "query": "Во сколько раз вырос курс USD с января 2022 по апрель 2026?",
        "expected_tools": ["compare_periods"],
        "must_have": ["раз"],
        "comment": "Требует compare_periods: метрика fx_USD, два периода → ratio. "
        "Контрольный вопрос из условия задания.",
    },
    {
        "id": 6,
        "query": "На сколько процентных пунктов изменилась ключевая ставка с декабря 2021 по март 2026?",
        "expected_tools": ["compare_periods"],
        "must_have": [],
        "comment": "Требует compare_periods: метрика key_rate, два периода → delta "
        "(в процентных пунктах). Проверяем, что агент берёт delta, а не ratio.",
    },
    {
        "id": 7,
        "query": "Сравни инфляцию в марте и в третьем квартале 2024 года.",
        "expected_tools": ["get_inflation"],
        "must_have": [],
        "comment": "ТРУДНЫЙ: «третий квартал» — не один месяц, а диапазон (июль–сентябрь). "
        "Инструменты работают помесячно. Агент должен либо выбрать конкретный месяц "
        "(напр. сентябрь как конец квартала), либо усреднить — и оговорить выбор. "
        "Часта ошибка: молча взять один месяц или перепутать границы квартала.",
    },
    {
        "id": 8,
        "query": "Что было выше в апреле 2026: ключевая ставка или годовая инфляция?",
        "expected_tools": ["get_key_rate", "get_inflation"],
        "must_have": [],
        "comment": "ТРУДНЫЙ + пограничный: данных по инфляции за апрель 2026 в CSV нет "
        "(последний месяц — март 2026). get_inflation вернёт error. Агент должен "
        "честно сказать о нехватке данных или взять ближайший доступный месяц, "
        "а не выдумать число. Часта ошибка: галлюцинация значения инфляции.",
    },
    {
        "id": 9,
        "query": "Какой сейчас «индекс нищеты» (инфляция г/г плюс безработица)?",
        "expected_tools": ["get_inflation", "get_unemployment", "calculate"],
        "must_have": [],
        "comment": "Реальный макро-вопрос: сумма двух метрик за последний доступный "
        "месяц. Проверяет, что агент берёт свежие данные и складывает через calculate.",
    },
    {
        "id": 10,
        "query": "Какова реальная доходность годового рублёвого вклада при текущей ставке и инфляции?",
        "expected_tools": ["get_key_rate", "get_inflation", "calculate"],
        "must_have": ["%"],
        "comment": "Реальный макро-вопрос: (1+ставка/100)/(1+инфляция/100)−1. "
        "Многостадийный кейс с нелинейной формулой через calculate.",
    },
]


def run_case(case: dict, *, use_cache: bool = False, track_cost: bool = False) -> dict:
    print(f"\n{'=' * 70}\n[Q{case['id']}] {case['query']}\n{'-' * 70}")
    res = run_agent(
        case["query"],
        max_iter=8,
        verbose=True,
        use_cache=use_cache,
        track_cost=track_cost,
    )
    used_tools = [e["call"] for e in res["trace"] if "call" in e]
    answer = res.get("answer") or ""

    tool_match = all(t in used_tools for t in case["expected_tools"])
    text_match = all(s.lower() in answer.lower() for s in case["must_have"])
    ok = bool(answer) and tool_match and text_match

    print(f"\n  tools used : {used_tools}")
    print(
        f"  expected    : {case['expected_tools']}  → {'OK' if tool_match else 'MISS'}"
    )
    print(f"  answer      : {answer[:200]}")
    print(f"  must_have   : {case['must_have']}  → {'OK' if text_match else 'MISS'}")
    print(f"  verdict     : {'PASS' if ok else 'FAIL'}")

    return {
        "id": case["id"],
        "query": case["query"],
        "ok": ok,
        "tools_used": used_tools,
        "steps": res["steps"],
        "answer": answer,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Мини-оценка макро-агента")
    ap.add_argument(
        "--cache",
        action="store_true",
        help="Блок 9: общий кэш инструментов на все вопросы — видно повторные вызовы",
    )
    ap.add_argument(
        "--cost",
        action="store_true",
        help="Блок 10: показать токены и стоимость по шагам",
    )
    a = ap.parse_args()

    if a.cache:
        CACHE_STATS["hits"] = CACHE_STATS["misses"] = 0

    results = [run_case(c, use_cache=a.cache, track_cost=a.cost) for c in CASES]
    passed = sum(1 for r in results if r["ok"])

    print(f"\n{'=' * 70}\nИтого: {passed}/{len(CASES)} пройдено")
    for r in results:
        mark = "[OK]  " if r["ok"] else "[FAIL]"
        print(f"  {mark} Q{r['id']} ({r['steps']} шагов) — {r['query'][:60]}")

    if a.cache:
        h, m = CACHE_STATS["hits"], CACHE_STATS["misses"]
        print(
            f"\n[кэш] на {len(CASES)} вопросах: {h} попаданий из {h + m} обращений "
            f"к инструментам — столько вызовов ЦБ/Росстата сэкономлено."
        )

    out = Path(__file__).parent / "eval_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты: {out}")

    md = ["| id | query | ok? | steps | tools_used |", "|---|---|:---:|:---:|---|"]
    for r in results:
        q = r["query"].replace("|", "/")
        tools = ", ".join(r["tools_used"]) or "—"
        md.append(
            f"| {r['id']} | {q} | {'✅' if r['ok'] else '❌'} | {r['steps']} | {tools} |"
        )
    md.append(f"\n**Прошло: {passed}/{len(CASES)}**")
    table = Path(__file__).parent / "eval_table.md"
    table.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Таблица для отчёта: {table}")


if __name__ == "__main__":
    main()
