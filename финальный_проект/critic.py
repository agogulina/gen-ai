"""
Скептик — второй агент мультиагентной связки. Получает оценку матчера и
ПЕРЕПРОВЕРЯЕТ её на завышение: типичная болезнь матчера — натянуть "Good Fit"
на слабые совпадения. Скептик видит требования, оценку и список доказательств
(уже проверенных верификатором), и может понизить вердикт.

Возвращает SkepticVerdict (структурированный вывод).
"""

from __future__ import annotations

from llm_client import get_model, make_client
from schema import FitAssessment, JobRequirements, SkepticVerdict

SKEPTIC_SYSTEM = """\
Ты — придирчивый второй рекрутёр (скептик). Тебе дают требования вакансии и
оценку соответствия от коллеги. Твоя задача — НЕ соглашаться из вежливости, а
проверить, не завышена ли оценка.

Считай оценку завышенной, если:
- "Good Fit" стоит при том, что ключевые обязательные навыки отсутствуют
  (есть в missing_skills) или подтверждены слабо;
- evidence малочисленны или не относятся к обязательным навыкам;
- confidence высокая, а совпадений мало.

ВАЖНО: не занижай искусственно. Если большинство обязательных навыков
подтверждено цитатами — соглашайся (agree=true). Понижай только при реальном
дефиците доказательств, а не «на всякий случай».

Верни:
- agree: согласен ли с оценкой коллеги;
- adjusted_fit: твоя итоговая метка ("No Fit"/"Potential Fit"/"Good Fit");
- reason: коротко, что вызвало сомнение или почему согласен.
"""


def skeptic_review(req: JobRequirements, assessment: FitAssessment,
                   *, temperature: float = 0.2) -> SkepticVerdict:
    client = make_client()
    body = (
        f"ТРЕБОВАНИЯ:\n- роль: {req.title}\n"
        f"- обязательные: {', '.join(req.must_have_skills) or '—'}\n\n"
        f"ОЦЕНКА КОЛЛЕГИ:\n- fit: {assessment.fit} (conf={assessment.confidence})\n"
        f"- matched: {', '.join(assessment.matched_skills) or '—'}\n"
        f"- missing: {', '.join(assessment.missing_skills) or '—'}\n"
        f"- доказательств: {len(assessment.evidence)}\n"
        f"- rationale: {assessment.rationale}"
    )
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": SKEPTIC_SYSTEM},
            {"role": "user", "content": body},
        ],
        response_model=SkepticVerdict,
        temperature=temperature,
        max_retries=2,
    )
