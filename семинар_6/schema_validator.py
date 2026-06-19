

from __future__ import annotations

from schemas_pwc import Plan

VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def validate_plan(plan: Plan) -> list[str]:
 
    errors: list[str] = []
    ids = [sq.id for sq in plan.subquestions]
    id_set = set(ids)

    if len(ids) != len(id_set):
        errors.append(f"дублирующиеся id подвопросов: {ids}")

    for sq in plan.subquestions:
        if not sq.expected_tools:
            errors.append(f"подвопрос {sq.id}: не указан ни один инструмент")
        for tool in sq.expected_tools:
            if tool not in VALID_TOOLS:
                errors.append(
                    f"подвопрос {sq.id}: инструмент '{tool}' не существует "
                    f"(разрешены {sorted(VALID_TOOLS)})"
                )
        for dep in sq.depends_on:
            if dep == sq.id:
                errors.append(f"подвопрос {sq.id}: зависит сам от себя")
            elif dep not in id_set:
                errors.append(
                    f"подвопрос {sq.id}: зависит от несуществующего id={dep}"
                )
    return errors


if __name__ == "__main__":
    from schemas_pwc import SubQuestion

    p1 = Plan(
        reasoning="CAGR курса доллара.",
        subquestions=[SubQuestion(id=1, question="CAGR USD?", expected_tools=["get_cagr"])],
    )
    p2 = Plan(
        reasoning="Прогноз и среднее.",
        subquestions=[
            SubQuestion(id=1, question="прогноз USD на 2027?", expected_tools=["get_forecast"]),
            SubQuestion(id=2, question="средняя ставка 2024?", expected_tools=["get_average"]),
            SubQuestion(id=3, question="сравнить", expected_tools=["calculate"], depends_on=[9]),
        ],
    )
    print("p1:", validate_plan(p1))
    print("p2:", validate_plan(p2))
