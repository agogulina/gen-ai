"""
RAG-ретривер по тексту резюме — Okapi BM25, написан руками (без LangChain и
без векторных БД). Идея: не пихать всё резюме (медиана ~5000 символов) в промпт,
а под каждое требование вакансии доставать только релевантные фрагменты.

Используется и как инструмент агента (search_resume), и для grounding-доказательств.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-zа-я0-9+#.]+", re.IGNORECASE)

# короткий стоп-лист, чтобы BM25 не уводило в общие слова
_STOP = {
    "the", "and", "for", "with", "you", "our", "are", "this", "that", "will",
    "have", "from", "your", "their", "была", "быть", "для", "что", "как", "или",
    "a", "an", "to", "of", "in", "on", "as", "is", "be", "or", "we", "by",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def chunk_text(text: str, *, max_chars: int = 320) -> list[str]:
    """Резюме → список фрагментов (по строкам/предложениям, с укрупнением)."""
    raw = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    chunks: list[str] = []
    buf = ""
    for piece in raw:
        piece = piece.strip()
        if not piece:
            continue
        if len(buf) + len(piece) + 1 <= max_chars:
            buf = f"{buf} {piece}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = piece[:max_chars] if len(piece) > max_chars else piece
    if buf:
        chunks.append(buf)
    return chunks or [(text or "").strip()[:max_chars]]


@dataclass
class Hit:
    chunk_id: int
    text: str
    score: float


class BM25:
    """Минимальный Okapi BM25 над набором фрагментов одного документа."""

    def __init__(self, chunks: list[str], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c) for c in chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.docs else 0.0
        self.freqs: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        for d in self.docs:
            tf: dict[str, int] = {}
            for tok in d:
                tf[tok] = tf.get(tok, 0) + 1
            self.freqs.append(tf)
            for tok in tf:
                self.df[tok] = self.df.get(tok, 0) + 1
        self.N = len(self.docs)

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        # idf со сглаживанием (как в классическом BM25), не уходим в минус
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query: str, top_k: int = 3) -> list[Hit]:
        q = tokenize(query)
        scored: list[Hit] = []
        for i, tf in enumerate(self.freqs):
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                idf = self._idf(term)
                f = tf[term]
                denom = f + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / (self.avgdl or 1)))
                s += idf * (f * (self.k1 + 1)) / (denom or 1)
            if s > 0:
                scored.append(Hit(chunk_id=i, text=self.chunks[i], score=round(s, 4)))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]


def build_index(resume_text: str) -> BM25:
    return BM25(chunk_text(resume_text))


if __name__ == "__main__":
    sample = ("Professional Summary. Worked on Selenium WebDriver and TestNG. "
              "Developed REST API tests in Python. Used Jenkins for CI. "
              "Knowledge of Apache POI and Page Object Model.")
    idx = build_index(sample)
    for h in idx.search("python api automation", top_k=2):
        print(round(h.score, 2), "::", h.text[:70])
