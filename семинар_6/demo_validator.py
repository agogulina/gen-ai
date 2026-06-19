

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schema_validator import validate_plan
from schemas_pwc import Plan, SubQuestion

OUT = Path(__file__).resolve().parent / "validator_demo.json"

PLAN_CAGR = Plan(
    reasoning="Посчитать среднегодовой темп роста курса доллара.",
    subquestions=[SubQuestion(id=1, question="CAGR курса USD за 2022-2026?", expected_tools=["get_cagr"])],
)

PLAN_FORECAST = Plan(
    reasoning="Спрогнозировать курс и усреднить ставку.",
    subquestions=[
        SubQuestion(id=1, question="прогноз курса USD на 2027?", expected_tools=["get_forecast"]),
        SubQuestion(id=2, question="средняя ставка за 2024?", expected_tools=["get_average"]),
        SubQuestion(id=3, question="сравнить", expected_tools=["calculate"], depends_on=[9]),
    ],
)


def run() -> dict:
    cases = []
    for name, plan in [("get_cagr_hallucination", PLAN_CAGR),
                       ("get_forecast_hallucination", PLAN_FORECAST)]:
        errs = validate_plan(plan)
        print(f"[{name}] {errs}")
        cases.append({
            "name": name,
            "reasoning": plan.reasoning,
            "subquestions": [
                {"id": s.id, "expected_tools": s.expected_tools, "depends_on": s.depends_on}
                for s in plan.subquestions
            ],
            "errors": errs,
        })
    payload = {"cases": cases}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено: {OUT}")
    return payload


if __name__ == "__main__":
    run()
