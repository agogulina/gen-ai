"""
Структурированный вывод системы матчинга «резюме ↔ вакансия».

Здесь сосредоточены бизнес-инварианты предметной области в виде
field_validator / model_validator — модель обязана возвращать данные,
которые им удовлетворяют (а llm_client делает max_retries при нарушении).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

FIT_LABELS = ("No Fit", "Potential Fit", "Good Fit")
FIT_ORDER = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}


class Seniority(str, Enum):
    junior = "junior"
    mid = "mid"
    senior = "senior"
    lead = "lead"
    unknown = "unknown"


# ---------------------------------------------------------------------------
# 1. Требования вакансии (извлекаются Планировщиком/экстрактором)
# ---------------------------------------------------------------------------
class JobRequirements(BaseModel):
    """Структурированные требования из текста вакансии."""

    title: str = Field(..., description="Краткое название роли")
    must_have_skills: list[str] = Field(
        default_factory=list, description="Ключевые обязательные навыки (3-10 коротких тегов)"
    )
    nice_to_have_skills: list[str] = Field(default_factory=list)
    min_years_experience: float = Field(
        0, description="Минимальный требуемый опыт в годах (0 если не указан)"
    )
    seniority: Seniority = Seniority.unknown

    @field_validator("min_years_experience")
    @classmethod
    def years_non_negative_and_sane(cls, v: float) -> float:
        # бизнес-инвариант: опыт не отрицательный и не абсурдный
        if v < 0:
            raise ValueError("min_years_experience не может быть отрицательным")
        if v > 60:
            raise ValueError("min_years_experience > 60 — явно ошибка извлечения")
        return float(v)

    @field_validator("must_have_skills", "nice_to_have_skills")
    @classmethod
    def normalize_skills(cls, v: list[str]) -> list[str]:
        # чистим пустые/дубликаты, приводим к нижнему регистру тегов
        out, seen = [], set()
        for s in v:
            s2 = " ".join(str(s).split()).strip()
            key = s2.lower()
            if s2 and key not in seen:
                seen.add(key)
                out.append(s2)
        return out


# ---------------------------------------------------------------------------
# 2. Единица доказательства: навык + ДОСЛОВНАЯ цитата из резюме
# ---------------------------------------------------------------------------
class Evidence(BaseModel):
    """Подтверждение навыка дословной цитатой из резюме (для ghost-проверки)."""

    skill: str = Field(..., description="Какой навык/требование подтверждает цитата")
    quote: str = Field(
        ..., description="ДОСЛОВНАЯ цитата из текста резюме (копировать точно, без пересказа)"
    )

    @field_validator("quote")
    @classmethod
    def quote_not_trivial(cls, v: str) -> str:
        v = " ".join(str(v).split()).strip()
        if len(v) < 2:
            raise ValueError("пустая цитата")
        return v[:400]  # длинное — обрезаем, а не отвергаем


# ---------------------------------------------------------------------------
# 3. Итоговая оценка соответствия
# ---------------------------------------------------------------------------
class FitAssessment(BaseModel):
    """Итог матчинга: метка соответствия + обоснование с доказательствами."""

    fit: Literal["No Fit", "Potential Fit", "Good Fit"]
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность 0..1")
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(
        default_factory=list, description="Доказательства для matched_skills"
    )
    rationale: str = Field(..., description="1-3 фразы: почему такая оценка")

    @field_validator("confidence")
    @classmethod
    def confidence_in_unit(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence должна быть в [0,1]")
        return float(v)

    @model_validator(mode="after")
    def good_fit_needs_evidence(self) -> "FitAssessment":
        # Бизнес-инвариант: вердикт "Good Fit" обязан опираться хотя бы на одно
        # подтверждённое доказательство и хотя бы один совпавший навык.
        if self.fit == "Good Fit":
            if not self.matched_skills:
                raise ValueError("Good Fit без matched_skills недопустим")
            if not self.evidence:
                raise ValueError("Good Fit обязан приводить evidence (цитаты)")
        return self


# ---------------------------------------------------------------------------
# 4. Вердикт скептика (мультиагент)
# ---------------------------------------------------------------------------
class SkepticVerdict(BaseModel):
    """Критик-скептик проверяет, не завышена ли оценка."""

    agree: bool = Field(..., description="True, если согласен с оценкой матчера")
    adjusted_fit: Literal["No Fit", "Potential Fit", "Good Fit"]
    reason: str = Field(..., description="Что именно вызвало сомнение / почему согласен")


# ---------------------------------------------------------------------------
# 5. Вердикт судьи (LLM-as-judge, используется в eval)
# ---------------------------------------------------------------------------
class JudgeVerdict(BaseModel):
    """Независимая оценка ОБОСНОВАННОСТИ ответа (не путать с правильностью метки)."""

    groundedness: int = Field(..., ge=1, le=5, description="1..5: насколько вывод опирается на резюме/вакансию")
    relevance: int = Field(..., ge=1, le=5, description="1..5: отвечает ли обоснование на вопрос соответствия")
    comment: str = ""

    @field_validator("groundedness", "relevance")
    @classmethod
    def in_1_5(cls, v: int) -> int:
        if not (1 <= int(v) <= 5):
            raise ValueError("оценка должна быть 1..5")
        return int(v)


if __name__ == "__main__":
    # быстрая проверка инвариантов
    ok = FitAssessment(fit="Good Fit", confidence=0.8, matched_skills=["python"],
                       evidence=[Evidence(skill="python", quote="Developed services in Python")],
                       rationale="strong overlap")
    print("valid Good Fit:", ok.fit)
    for bad in [
        dict(fit="Good Fit", confidence=0.9, matched_skills=[], evidence=[], rationale="x"),
        dict(fit="No Fit", confidence=1.5, rationale="x"),
    ]:
        try:
            FitAssessment(**bad)
            print("НЕ поймал:", bad)
        except Exception as e:
            print("поймал инвариант:", str(e)[:60])
