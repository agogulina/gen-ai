"""
Матчер: агент с инструментами + структурированный вывод.

Поток:
  1. extract_requirements() — структурно вынуть требования вакансии (JobRequirements).
  2. run_matcher_agent() — ReAct-цикл: модель сама зовёт search_resume /
     check_skill_present, собирая доказательства по требованиям.
  3. финализация — из накопленного транскрипта получить FitAssessment
     (структурированный вывод с валидаторами, max_retries).

Используются и raw-клиент (tool calling), и JSON-клиент (структурный вывод) —
оба из вашего llm_client.py (DeepSeek).
"""

from __future__ import annotations

import json
from typing import Any

from llm_client import get_model, make_client, make_raw_client
from schema import FitAssessment, JobRequirements
from tools import TOOL_SCHEMAS, make_tools

REQ_SYSTEM = (
    "Ты — рекрутёр-аналитик. Извлеки из текста вакансии структурированные "
    "требования: короткое название роли, обязательные навыки (3-10 тегов), "
    "желательные навыки, минимальный опыт в годах (0 если не указан), уровень "
    "(junior/mid/senior/lead/unknown). Навыки — короткими тегами, без воды."
)

MATCH_SYSTEM = """\
Ты — технический рекрутёр. Оцени соответствие кандидата вакансии, опираясь
ТОЛЬКО на факты из резюме. У тебя есть инструменты:
- search_resume(query): найти релевантные фрагменты резюме;
- check_skill_present(skill): проверить, упомянут ли навык дословно.

Алгоритм:
1. По каждому обязательному навыку вакансии вызови search_resume и/или
   check_skill_present, чтобы найти подтверждение В РЕЗЮМЕ.
2. НЕ придумывай опыт, которого нет во фрагментах. Если навыка нет — он missing.
3. Когда собрал достаточно фактов, ответь обычным текстом БЕЗ вызова инструментов
   словом DONE и краткой выжимкой: какие навыки подтверждены (с дословной цитатой
   из фрагмента) и каких не хватает.
"""

FINAL_SYSTEM = """\
Ты выносишь финальную оценку ОБЩЕГО соответствия кандидата вакансии — по
профессии/домену, уровню и переносимому опыту, а не по дословному совпадению
каждого навыка. Частичное совпадение навыков — норма для реальных резюме.

Тебе дан сигнал ПОКРЫТИЯ обязательных навыков (какие найдены в резюме, какие нет)
и транскрипт поиска — опирайся на них.

Калибровка метки:
- "Good Fit": кандидат явно из этой профессии и закрывает большинство ключевых
  требований (покрытие высокое);
- "Potential Fit": частичное совпадение — смежная роль / часть навыков / есть
  переносимый опыт (среднее покрытие);
- "No Fit": другая профессия и почти нет пересечения (покрытие очень низкое).

ПРАВИЛА ЗАПОЛНЕНИЯ:
- Каждый подтверждённый навык ОБЯЗАН попасть в matched_skills (как минимум все из
  списка «найдены»), с ДОСЛОВНОЙ цитатой из резюме в evidence.
- missing_skills — только реально отсутствующие.
- "Good Fit" требует непустых matched_skills и evidence.
- Это инструмент СКРИНИНГА: при сомнении склоняйся к "Potential Fit", а не к
  "No Fit". Ставь "No Fit" только когда кандидат явно из ДРУГОЙ профессии и
  переносимого опыта практически нет.
Заполни: fit, confidence 0..1, matched_skills, missing_skills, evidence, rationale.
"""

MAX_STEPS = 5


def extract_requirements(job_description: str, *, temperature: float = 0.0) -> JobRequirements:
    client = make_client()
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": REQ_SYSTEM},
            {"role": "user", "content": job_description[:6000]},
        ],
        response_model=JobRequirements,
        temperature=temperature,
        max_retries=2,
    )


def _run_react(resume_text: str, req: JobRequirements) -> tuple[str, list[dict], list[dict]]:
    """ReAct-цикл по инструментам. Возвращает (transcript_text, tool_log, raw_trace)."""
    impl, used = make_tools(resume_text)
    client = make_raw_client()
    model = get_model()

    req_brief = (
        f"Вакансия: {req.title}\n"
        f"Обязательные навыки: {', '.join(req.must_have_skills) or '—'}\n"
        f"Желательные: {', '.join(req.nice_to_have_skills) or '—'}\n"
        f"Мин. опыт: {req.min_years_experience} лет; уровень: {req.seniority.value}"
    )

    # Предварительный RAG: по каждому обязательному навыку достаём фрагменты резюме,
    # чтобы модель не объявляла навык "missing" просто потому, что сама не искала.
    pre_lines = []
    for sk in req.must_have_skills[:8]:
        r = impl["search_resume"](sk, top_k=2)
        hits = r.get("hits") or []
        best = hits[0]["text"][:200] if hits and hits[0].get("text") else "—"
        pre_lines.append(f"- «{sk}»: {best}")
    pre_block = "\n".join(pre_lines) or "(требования без явных навыков)"

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": MATCH_SYSTEM},
        {"role": "user", "content": (
            req_brief
            + "\n\nПредварительно найденные фрагменты резюме по каждому навыку:\n"
            + pre_block
            + "\n\nПроверь их, при необходимости уточни поиск инструментами и "
              "сделай вывод по каждому обязательному навыку (подтверждён/нет)."
        )},
    ]
    trace: list[dict] = []

    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SCHEMAS,
            tool_choice="auto", temperature=0.0,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return (msg.content or ""), used, trace
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            fn = impl.get(name)
            obs = fn(**args) if fn else {"error": f"unknown tool {name}"}
            trace.append({"call": name, "args": args, "obs": obs})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(obs, ensure_ascii=False)[:1500]})
    # лимит шагов — просим финальную выжимку без инструментов
    messages.append({"role": "user", "content": "Достаточно. Дай выжимку (DONE) без инструментов."})
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
    return (resp.choices[0].message.content or ""), used, trace


def compute_coverage(resume_text: str, req: JobRequirements) -> dict:
    """Детерминированно посчитать покрытие обязательных навыков резюме.

    Навык считается «найденным», если он сам ИЛИ его синоним/вариант написания
    встречается в резюме (skills.skill_present, с нормализацией и словарём
    синонимов) ИЛИ имеет заметный BM25-хит. Синонимы поднимают recall на
    профильных резюме, где навык записан другими словами.
    """
    from retrieval import build_index
    from skills import is_generic, normalize_resume, skill_present

    rnorm = normalize_resume(resume_text)
    idx = build_index(resume_text)
    present, absent = [], []
    # дискриминативные навыки (без слишком общих) — по ним и считаем покрытие
    disc = [sk for sk in req.must_have_skills if not is_generic(sk)] or req.must_have_skills
    for sk in disc:
        hit = skill_present(sk, rnorm) or any(h.score >= 1.2 for h in idx.search(sk, top_k=1))
        (present if hit else absent).append(sk)
    total = len(disc)
    ratio = (len(present) / total) if total else 0.0
    return {"present": present, "absent": absent, "total": total, "ratio": round(ratio, 3)}


# --- Пороги решения по покрытию (легко крутить под свои данные) ---
# Граница Fit/No Fit (в бинарном режиме) = FIT_THRESHOLD.
GOOD_THRESHOLD = 0.50   # >= -> Good Fit
FIT_THRESHOLD = 0.30    # >= -> Potential Fit (и = граница No Fit/Fit в бинаре)


def _coverage_label(ratio: float, has_reqs: bool) -> str:
    """Калиброванная метка по покрытию. Пороги вынесены в константы выше."""
    if not has_reqs:
        return "Potential Fit"
    if ratio >= GOOD_THRESHOLD:
        return "Good Fit"
    if ratio >= FIT_THRESHOLD:
        return "Potential Fit"
    return "No Fit"


def run_matcher_agent(resume_text: str, job_description: str) -> dict:
    """Полный матчинг одной пары. Возвращает dict с оценкой и метриками пути."""
    req = extract_requirements(job_description)
    coverage = compute_coverage(resume_text, req)
    transcript, used, trace = _run_react(resume_text, req)

    client = make_client()
    final_user = (
        f"ТРЕБОВАНИЯ ВАКАНСИИ:\n{req.model_dump()}\n\n"
        f"ПОКРЫТИЕ ОБЯЗАТЕЛЬНЫХ НАВЫКОВ (объективный сигнал):\n"
        f"- найдены в резюме: {coverage['present'] or '—'}\n"
        f"- не найдены: {coverage['absent'] or '—'}\n"
        f"- доля покрытия: {coverage['ratio']}\n\n"
        f"ТРАНСКРИПТ ПОИСКА ПО РЕЗЮМЕ (факты и цитаты):\n{transcript[:4000]}"
    )
    assessment = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": FINAL_SYSTEM},
            {"role": "user", "content": final_user},
        ],
        response_model=FitAssessment,
        temperature=0.0,
        max_retries=2,
    )
    return {
        "requirements": req,
        "assessment": assessment,
        "coverage": coverage,
        "coverage_label": _coverage_label(coverage["ratio"], coverage["total"] > 0),
        "tool_log": used,
        "trace": trace,
        "n_tool_calls": len(used),
        "n_steps": len(trace) + 1,
    }


if __name__ == "__main__":
    import sys
    rt = "Developed REST API tests in Python. Used Jenkins for CI. Selenium WebDriver, TestNG."
    jd = "Looking for QA Automation Engineer with Python, Selenium, CI/CD (Jenkins). 2+ years."
    out = run_matcher_agent(rt, jd)
    print(out["assessment"].model_dump())
