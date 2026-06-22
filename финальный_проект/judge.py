"""
LLM-as-judge: независимая оценка ОБОСНОВАННОСТИ вывода системы.

Важно: судья оценивает не «угадана ли метка» (это считаем по gold-лейблу
датасета), а насколько обоснование опирается на факты резюме/вакансии и
отвечает по существу. Это ортогональная метрика «пути/качества».
"""

from __future__ import annotations

from llm_client import get_model, make_client
from schema import JudgeVerdict

JUDGE_SYSTEM = """\
Ты — старший рекрутёр-ревьюер. Оцени КАЧЕСТВО ОБОСНОВАНИЯ оценки соответствия
(не саму метку). Даны: требования вакансии, вывод системы (метка, совпавшие/
недостающие навыки, цитаты-доказательства, rationale).

Поставь две оценки по шкале 1..5:
- groundedness: насколько вывод опирается на конкретные факты из резюме
  (цитаты к месту, нет общих слов и выдумок);
- relevance: отвечает ли обоснование на вопрос соответствия именно этим
  требованиям.
Дай короткий comment.
"""


def judge_assessment(requirements: dict, result: dict, *, temperature: float = 0.0) -> JudgeVerdict:
    client = make_client()
    body = (
        f"ТРЕБОВАНИЯ: {requirements}\n\n"
        f"ВЫВОД СИСТЕМЫ:\n"
        f"- метка: {result.get('final_fit')}\n"
        f"- matched: {result.get('matched_skills')}\n"
        f"- missing: {result.get('missing_skills')}\n"
        f"- доказательства: {result.get('evidence')}\n"
        f"- rationale: {result.get('rationale')}"
    )
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": body},
        ],
        response_model=JudgeVerdict,
        temperature=temperature,
        max_retries=2,
    )
