"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик.

На семинаре нужно:
- реализовать topological_sort (TODO 1),
- реализовать replan/rework-ветки цикла (TODO 2),
- написать synthesize для финального ответа (TODO 3).

Важно: max_iter защищает от бесконечного цикла, если Критик
постоянно говорит «переделай».
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from concurrent.futures import ThreadPoolExecutor

from critic import critic
from llm_client import get_model, make_client, make_raw_client
from planner import planner
from schema_validator import VALID_TOOLS, validate_plan
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from worker import worker


def _topological_sort(subqs: list[SubQuestion]) -> list[SubQuestion]:
    """Отсортировать подвопросы так, чтобы depends_on шли раньше (DFS)."""
    by_id = {s.id: s for s in subqs}
    ordered: list[SubQuestion] = []
    visited: set[int] = set()

    def visit(node_id: int, path: list[int]):
        if node_id in visited:
            return None
        if node_id in path:
            raise ValueError(f"Цикл в depends_on: {path + [node_id]}")
        if node_id not in by_id:
            return None  # висячая ссылка — тихо пропускаем
        for dep in by_id[node_id].depends_on:
            visit(dep, path + [node_id])
        visited.add(node_id)
        ordered.append(by_id[node_id])

    for sq in subqs:
        visit(sq.id, [])
    return ordered


def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    """ДЗ С6, часть 2: сгруппировать подвопросы по УРОВНЯМ зависимостей.

    Между уровнями есть зависимость, внутри уровня — нет (всё независимо и
    может исполняться параллельно). Замена плоского _topological_sort.
    """
    by_id = {s.id: s for s in subqs}
    resolved: set[int] = set()
    levels: list[list[SubQuestion]] = []
    remaining = list(subqs)

    while remaining:
        # уровень = подвопросы, все зависимости которых уже решены
        # (висячие ссылки на несуществующие id игнорируем — их некого ждать)
        level = [
            sq for sq in remaining
            if all(dep in resolved or dep not in by_id for dep in sq.depends_on)
        ]
        if not level:  # цикл/тупик — кладём остаток одним уровнем, чтобы не зависнуть
            level = remaining
        levels.append(level)
        for sq in level:
            resolved.add(sq.id)
        remaining = [sq for sq in remaining if sq.id not in resolved]
    return levels


def execute_level(
    level: list[SubQuestion], prev_answers: dict[int, WorkerAnswer]
) -> dict[int, WorkerAnswer]:
    """ДЗ С6, часть 2: прогнать все подвопросы уровня ПАРАЛЛЕЛЬНО."""
    out: dict[int, WorkerAnswer] = {}
    if not level:
        return out
    with ThreadPoolExecutor(max_workers=len(level)) as pool:
        futures = {pool.submit(worker, sq, prev_answers): sq for sq in level}
        for fut, sq in futures.items():
            out[sq.id] = fut.result()
    return out


def execute_level_sequential(
    level: list[SubQuestion], prev_answers: dict[int, WorkerAnswer]
) -> dict[int, WorkerAnswer]:
    """Та же логика, но строго по одному — для замера ускорения (часть 2)."""
    out: dict[int, WorkerAnswer] = {}
    for sq in level:
        out[sq.id] = worker(sq, prev_answers)
    return out


_SYNTH_SYSTEM = (
    "Ты собираешь финальный ответ пользователю из ответов на подвопросы. "
    "Используй ТОЛЬКО приведённые числа, ничего не пересчитывай в уме. "
    "Дай 1-2 фразы с числами и единицами измерения."
)


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    """Собрать финальный ответ одним LLM-вызовом без tools (ДЗ блок 3.3)."""
    if not answers:
        return plan.reasoning or "(нет ответов для синтеза)"
    parts = [
        f"{a.subquestion_id}. {a.answer}" for a in
        (answers[i] for i in sorted(answers))
    ]
    body = f"Вопрос пользователя: {question}\n\nОтветы на подвопросы:\n" + "\n".join(parts)
    try:
        client = make_raw_client()
        resp = client.chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": _SYNTH_SYSTEM},
                {"role": "user", "content": body},
            ],
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or " · ".join(a.answer for a in answers.values())
    except Exception:
        # если синтез по сети не удался — не падаем, склеиваем ответы
        return " · ".join(answers[i].answer for i in sorted(answers))


def _validate_and_replan(question, plan, trace, *, max_replans, verbose):
    """ДЗ С6, часть 1: валидатор схемы сразу после planner().

    Если есть ошибки — зовём planner(question, feedback=...) и так до
    max_replans раз. Если и после этого план битый — отбрасываем подвопросы
    с несуществующими инструментами (может остаться пустой план — это ок,
    значит задача нерешаема доступными tools).
    """
    for _ in range(max_replans):
        errors = validate_plan(plan)
        trace.append({"kind": "validate", "errors": errors})
        if not errors:
            return plan
        if verbose:
            print(f"  [validator] ошибки плана: {errors}")
        plan = planner(question, feedback=f"Инструменты не существуют: {errors}")
        trace.append(
            {
                "kind": "replan_after_validate",
                "reasoning": plan.reasoning,
                "subquestions": [sq.model_dump() for sq in plan.subquestions],
            }
        )
    # последняя попытка всё ещё битая — выкидываем подвопросы с фейк-инструментами
    plan.subquestions = [
        sq for sq in plan.subquestions if not (set(sq.expected_tools) - VALID_TOOLS)
    ]
    return plan


def run_pwc(
    question: str,
    *,
    max_iter: int = 3,
    verbose: bool = True,
    use_validator: bool = True,
    parallel: bool = True,
    max_replans: int = 2,
    critic_temperature: float = 0.7,
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик.

    use_validator — включить валидатор схемы (ДЗ С6, часть 1);
    parallel       — исполнять независимые подвопросы уровня параллельно (часть 2).
    """
    trace: list[dict[str, Any]] = []
    answers: dict[int, WorkerAnswer] = {}

    plan = planner(question)
    trace.append(
        {
            "iter": 0,
            "kind": "plan",
            "reasoning": plan.reasoning,
            "subquestions": [sq.model_dump() for sq in plan.subquestions],
        }
    )

    # --- ДЗ С6, часть 1: валидатор схемы между Планировщиком и Исполнителем ---
    if use_validator:
        plan = _validate_and_replan(
            question, plan, trace, max_replans=max_replans, verbose=verbose
        )

    if verbose:
        print(f"\n[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")

    if not plan.subquestions:
        msg = (
            "Такую задачу нельзя решить доступными инструментами "
            f"({sorted(VALID_TOOLS)}). {plan.reasoning}"
        )
        return {
            "answer": msg,
            "plan": plan,
            "answers": {},
            "trace": trace,
            "iterations": 0,
        }

    runner = execute_level if parallel else execute_level_sequential

    for iter_num in range(1, max_iter + 1):
        answers = {}
        # --- ДЗ С6, часть 2: исполнение по уровням зависимостей ---
        for level in _topological_levels(plan.subquestions):
            level_answers = runner(level, answers)
            answers.update(level_answers)
            for sq in level:
                ans = answers[sq.id]
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "worker",
                        "sq_id": sq.id,
                        "used_tools": ans.used_tools,
                        "answer": ans.answer,
                    }
                )
                if verbose:
                    print(f"  [{sq.id}] → {ans.answer}   tools={ans.used_tools}")

        verdict = critic(question, plan, answers, temperature=critic_temperature)
        trace.append(
            {
                "iter": iter_num,
                "kind": "verdict",
                "ok": verdict.ok,
                "action": verdict.action,
                "reason": verdict.reason,
                "rework_ids": verdict.rework_ids,
            }
        )

        if verbose:
            mark = "OK" if verdict.ok else "FAIL"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        if verdict.ok:
            final = _synthesize(question, plan, answers)
            return {
                "answer": final,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        # --- ДЗ блок 3.2: реакция на вердикт Критика ---
        if iter_num == max_iter:
            break
        if verdict.action == "rework":
            fb = f"Переделай подвопросы {verdict.rework_ids}: {verdict.reason}"
        else:  # replan / прочее
            fb = verdict.reason
        plan = planner(question, feedback=fb)
        if use_validator:
            plan = _validate_and_replan(
                question, plan, trace, max_replans=max_replans, verbose=verbose
            )
        if not plan.subquestions:
            break

    return {
        "answer": _synthesize(question, plan, answers) if answers else None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-validator", action="store_true",
                    help="Отключить валидатор схемы (ДЗ С6, часть 1)")
    ap.add_argument("--sequential", action="store_true",
                    help="Исполнять подвопросы по одному (без параллельности)")
    ap.add_argument(
        "--trace", type=Path, default=None, help="Куда сохранить JSON-лог (если задан)"
    )
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_pwc(
        q,
        max_iter=args.max_iter,
        verbose=not args.quiet,
        use_validator=not args.no_validator,
        parallel=not args.sequential,
    )

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')})")

    if args.trace:
        args.trace.write_text(
            json.dumps(
                {"query": q, **_serialize(res)},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
