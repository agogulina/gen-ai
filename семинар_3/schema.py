"""
schema.py — Pydantic-схемы пайплайна (предметная область: отзывы на приложение)
================================================================================

Карта моделей по раундам:
  Раунд 1   — Issue, Review                  
  Раунд 2   — AspectSentiment, ReviewAspects  
  Раунд 2.5 — DiscoveredAspect/-s, Dynamic* 
  Раунд 3   — ChunkSummary, AppSummary         
  Раунд 5   — ActionVerdict, JudgeReport       
"""
from __future__ import annotations

from datetime import date as _date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Вариант A
ASPECTS = ["performance", "design", "support", "price", "ads", "reliability"]
ASPECT_RU = {
    "performance": "производительность",
    "design": "дизайн",
    "support": "поддержка",
    "price": "цена",
    "ads": "реклама",
    "reliability": "надёжность",
}

AspectName = Literal["performance", "design", "support", "price", "ads", "reliability"]
Sentiment = Literal["positive", "negative", "neutral"]



# Раунд 1 — Information Extraction
class Issue(BaseModel):
 
    category: Literal[
        "performance", "design", "support", "price", "ads", "reliability", "other"
    ]
    severity: int = Field(ge=1, le=5, description="1 — мелочь, 5 — блокер")
    quote: str = Field(min_length=1, description="дословная цитата из отзыва")

    @field_validator("quote")
    @classmethod
    def _quote_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("quote не может быть пустой")
        return v


class Review(BaseModel):
    """Один отзыв с извлечёнными проблемами."""

    review_id: str
    author: Optional[str] = None
    rating: int = Field(ge=1, le=5)
    date: Optional[str] = None
    app_version: Optional[str] = None
    issues: list[Issue]
    mentioned_apps: list[str] = Field(
        default_factory=list, description="упомянутые сторонние приложения/конкуренты"
    )

    @field_validator("date")
    @classmethod
    def _date_not_in_future(cls, v: Optional[str]) -> Optional[str]:
        """дата отзыва не позже сегодняшней"""
        if v is None or not v.strip():
            return None
        try:
            d = datetime.strptime(v.strip(), "%Y-%m-%d").date()
        except ValueError as e:
            raise ValueError(f"дата должна быть в формате YYYY-MM-DD, получено {v!r}") from e
        if d > _date.today():
            raise ValueError(f"дата отзыва {v} в будущем — это невозможно")
        return v.strip()


class MatchVerdict(BaseModel):
    matched: bool
    matched_index: int = Field(default=-1, description="номер проблемы или -1")
    reason: str = ""


# Раунд 2 — Аспектный анализ

class AspectSentiment(BaseModel):
    aspect: AspectName
    sentiment: Sentiment
    quote: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @field_validator("quote")
    @classmethod
    def _quote_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("quote не может быть пустой")
        return v


class ReviewAspects(BaseModel):
    review_id: str
    aspects: list[AspectSentiment]


# Autodiscovery аспектов
class DiscoveredAspect(BaseModel):
    name: str
    description: str = Field(min_length=5)


class DiscoveredAspects(BaseModel):
    aspects: list[DiscoveredAspect] = Field(min_length=3, max_length=12)


class DynamicAspect(BaseModel):
    aspect: str
    sentiment: Sentiment
    quote: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class DynamicReview(BaseModel):
    review_id: str
    aspects: list[DynamicAspect]


# Раунд 3 — Map-Reduce-резюме
class ChunkSummary(BaseModel):

    covers: list[str] = Field(min_length=1, description="review_id отзывов в пачке")
    key_points: list[str] = Field(min_length=1, max_length=6)
    sentiment: Literal["positive", "negative", "mixed"]


class AppSummary(BaseModel):

    headline: str
    key_findings: list[str] = Field(min_length=2, max_length=8)
    action_items: list[str] = Field(min_length=1, max_length=8)


# Раунд 5 — LLM-as-judge
class ActionVerdict(BaseModel):
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict]
    overall_score: float = Field(ge=0, le=1)
    summary: str
