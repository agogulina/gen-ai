"""
стратегии чанкинга:
  fixed      — text[i:i+2000], без перекрытия (Стратегия A).
  recursive  — рекурсивное разбиение по абзацам/предложениям,
               chunk_size=400, overlap=80 (Стратегия B).
"""

from __future__ import annotations

import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
BACKEND = os.environ.get("RAG_BACKEND", "auto").lower()


# Токенизация
def tokenize_ru(text: str) -> list[str]:
    return re.findall(r"[а-яa-zё0-9+#.-]{2,}", text.lower())


# стратегии чанкинга
def chunk_fixed(text: str, chunk_size: int = 2000) -> list[str]:
    # Стратегия A
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if len(text) <= chunk_size or not separators:
        return [text]
    sep = separators[0]
    parts = text.split(sep) if sep else list(text)
    out: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            out.append(part)
        else:
            out.extend(_recursive_split(part, separators[1:], chunk_size))
    return out


def chunk_recursive(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    # Стратегия B
    seps = ["\n\n", "\n", ". ", "? ", "! ", " "]
    pieces = _recursive_split(text, seps, chunk_size)
    chunks: list[str] = []
    cur = ""
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= chunk_size:
            cur = cur + " " + p
        else:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + " " + p).strip()
    if cur:
        chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]


def get_chunker(strategy: str):
    if strategy == "fixed":
        return chunk_fixed
    if strategy == "recursive":
        return chunk_recursive
    raise ValueError(f"Неизвестная стратегия чанкинга: {strategy}")


class BM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = corpus_tokens
        self.k1, self.b = k1, b
        self.N = max(1, len(corpus_tokens))
        self.avgdl = sum(len(d) for d in corpus_tokens) / self.N
        df: dict[str, int] = {}
        for d in corpus_tokens:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }
        self.tf = [Counter(d) for d in corpus_tokens]

    def scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.N
        for t in query_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if not f:
                    continue
                dl = len(self.docs[i])
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


class TfidfDense:

    name = "tfidf"

    def __init__(self, texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(tokenizer=tokenize_ru, lowercase=False, token_pattern=None)
        self.matrix = self.vec.fit_transform(texts)  # уже L2-нормирован

    def scores(self, query: str) -> list[float]:
        from sklearn.metrics.pairwise import linear_kernel

        q = self.vec.transform([query])
        return linear_kernel(q, self.matrix)[0].tolist()


class SbertDense:

    name = "sbert"

    def __init__(self, texts: list[str]):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.emb = self.model.encode(texts, normalize_embeddings=True)

    def scores(self, query: str) -> list[float]:
        import numpy as np

        q = self.model.encode([query], normalize_embeddings=True)[0]
        return (self.emb @ q).tolist()


def make_dense(texts: list[str]):
    if BACKEND == "tfidf":
        return TfidfDense(texts)
    if BACKEND == "sbert":
        return SbertDense(texts)
    # auto
    try:
        return SbertDense(texts)
    except Exception:
        return TfidfDense(texts)

class Index:
    def __init__(self, chunk_strategy: str):
        self.strategy = chunk_strategy
        chunker = get_chunker(chunk_strategy)
        self.ids: list[str] = []
        self.texts: list[str] = []
        for f in sorted(DATA_DIR.glob("*.txt")):
            for i, c in enumerate(chunker(f.read_text(encoding="utf-8"))):
                self.ids.append(f"{f.stem}__{i}")
                self.texts.append(c)
        self.bm25 = BM25([tokenize_ru(t) for t in self.texts])
        self.dense = make_dense(self.texts)

    def stats(self) -> str:
        per: dict[str, int] = {}
        for cid in self.ids:
            src = cid.split("__")[0]
            per[src] = per.get(src, 0) + 1
        avg = sum(len(t) for t in self.texts) / max(1, len(self.texts))
        return (
            f"[{self.strategy:9s} | dense={self.dense.name}] "
            f"чанков: {len(self.ids)} из {len(per)} документов, "
            f"средняя длина чанка {avg:.0f} симв."
        )

    def retrieve(self, query: str, k: int = 5, top: int = 20, c: int = 60) -> dict:
        dense_scores = self.dense.scores(query)
        sparse_scores = self.bm25.scores(tokenize_ru(query))

        order = lambda s: sorted(range(len(s)), key=lambda i: s[i], reverse=True)[:top]
        dense_rank = order(dense_scores)
        sparse_rank = order(sparse_scores)

        rrf: dict[int, float] = {}
        for rank, idx in enumerate(dense_rank):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (c + rank)
        for rank, idx in enumerate(sparse_rank):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (c + rank)

        topk = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
        ids = [self.ids[i] for i, _ in topk]
        docs = [self.texts[i] for i, _ in topk]
        return {"ids": [ids], "documents": [docs]}


def build_prompt(query: str, hits: dict) -> str:
    docs, ids = hits["documents"][0], hits["ids"][0]
    ctx = "\n\n---\n\n".join(f"[{i}]\n{d}" for i, d in zip(ids, docs))
    return (
        "Ты отвечаешь на вопрос по корпусу статей о языках программирования.\n"
        "Правила:\n"
        "1. Опирайся ТОЛЬКО на контекст ниже. Не добавляй факты из общего знания.\n"
        "2. В `quotes` — 1-5 точных коротких цитат из контекста.\n"
        "3. В `sources` — id блоков, откуда взято (формат: 'lang_rust__0').\n"
        "4. В `confidence` — 0.9+ только при прямом ответе; 0.5-0.8 если собран "
        "из нескольких кусков; < 0.5 если контекст не отвечает.\n\n"
        f"Контекст:\n{ctx}\n\nВопрос: {query}\n\nОтвет:"
    )


def ask(query: str):
    strategy = os.environ.get("CHUNK_STRATEGY", "recursive")
    idx = Index(strategy)
    print(idx.stats(), flush=True)
    hits = idx.retrieve(query, k=5)
    print("Найдены чанки:", ", ".join(hits["ids"][0]), flush=True)

    try:
        from llm_client import get_model, make_client
        from schema import RAGAnswer

        client, model = make_client(), get_model()
        resp: RAGAnswer = client.chat.completions.create(
            model=model,
            response_model=RAGAnswer,
            messages=[{"role": "user", "content": build_prompt(query, hits)}],
            temperature=0.2,
        )
        print("\n" + "=" * 60)
        print(f"ВОПРОС: {query}\n" + "=" * 60)
        print(resp)
    except Exception as e:
        print(
            f"\n[Генерация ответа пропущена: {e}]\n"
            "Чтобы получить ответ LLM, настройте .env"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python pipeline.py {ingest|ask} [...]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "ingest":
        strat = sys.argv[2] if len(sys.argv) > 2 else "recursive"
        print(Index(strat).stats())
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print('Нужен вопрос: python pipeline.py ask "..."')
            sys.exit(1)
        ask(sys.argv[2])
    else:
        print(f"Неизвестная команда: {cmd}")
        sys.exit(1)
