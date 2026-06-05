
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from llm_client import get_model, make_client
from prompts import (
    ASPECTS_SYSTEM,
    CHUNK_SYSTEM,
    DISCOVER_SYSTEM,
    IE_SYSTEM,
    JUDGE_SYSTEM,
    REDUCE_SYSTEM,
    REDUCE_SYSTEM_STRICT,
)
from schema import (
    ASPECT_RU,
    ASPECTS,
    AppSummary,
    ChunkSummary,
    DiscoveredAspects,
    DynamicReview,
    JudgeReport,
    Review,
    ReviewAspects,
)

client = make_client()
MODEL = get_model()

# Цены за 1M токенов (USD)
PRICE_IN = float(os.environ.get("LLM_PRICE_INPUT", "0.14"))
PRICE_OUT = float(os.environ.get("LLM_PRICE_OUTPUT", "0.28"))

# Глобальный аккумулятор расхода токенов по всему прогону.
USAGE = {"prompt": 0, "completion": 0, "cache_hit": 0, "calls": 0}


def call(response_model, system: str, user: str, max_retries: int = 3):
    result, completion = client.chat.completions.create(
        model=MODEL,
        response_model=response_model,
        max_retries=max_retries,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        with_completion=True,
    )
    u = completion.usage
    USAGE["prompt"] += getattr(u, "prompt_tokens", 0) or 0
    USAGE["completion"] += getattr(u, "completion_tokens", 0) or 0
    USAGE["cache_hit"] += getattr(u, "prompt_cache_hit_tokens", 0) or 0
    USAGE["calls"] += 1
    return result


def usage_cost() -> dict:
    cost = USAGE["prompt"] / 1e6 * PRICE_IN + USAGE["completion"] / 1e6 * PRICE_OUT
    return {
        "calls": USAGE["calls"],
        "prompt_tokens": USAGE["prompt"],
        "completion_tokens": USAGE["completion"],
        "cache_hit_tokens": USAGE["cache_hit"],
        "cost_usd": round(cost, 4),
    }

def load_reviews(input_path: str) -> list[dict]:
    p = Path(input_path)
    if p.is_dir():
        p = p / "reviews.csv"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    for r in rows:
        r["rating"] = int(r["rating"])
        r["thumbs_up"] = int(r.get("thumbs_up", 0) or 0)
    return rows


def review_block(r: dict) -> str:
    return (
        f"[review_id={r['review_id']} | author={r['author']} | "
        f"rating={r['rating']} | date={r['date']}]\n{r['text']}"
    )


def build_corpus(rows: list[dict]) -> str:
    return "\n\n".join(review_block(r) for r in rows)


def raw_text(rows: list[dict]) -> str:
    """Чистый текст всех отзывов (для подстрочной проверки цитат)."""
    return "\n".join(r["text"] for r in rows).lower()


def batched(rows: list[dict], size: int):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


# Раунд 1 — IE
def extract_reviews(rows: list[dict], batch_size: int = 6) -> tuple[list[Review], dict]:

    extracted: list[Review] = []
    failed_ids: list[str] = []
    for batch in batched(rows, batch_size):
        corpus = build_corpus(batch)
        try:
            res = call(list[Review], IE_SYSTEM, corpus)
            extracted.extend(res)
        except Exception as e:  # noqa: BLE001 — нам важно не упасть, а посчитать
            print(f"батч {[r['review_id'] for r in batch]} не прошёл валидацию: {e}")
            failed_ids.extend(r["review_id"] for r in batch)

    got_ids = {r.review_id for r in extracted}
    expected_ids = {r["review_id"] for r in rows}
    dropped = sorted(expected_ids - got_ids - set(failed_ids))
    stats = {
        "expected": len(rows),
        "valid": len(extracted),
        "failed_validation": len(failed_ids),
        "failed_ids": failed_ids,
        "dropped_by_model": dropped,
    }
    return extracted, stats


# Раунд 2 — Аспектный анализ
def extract_aspects(rows: list[dict], batch_size: int = 6) -> list[ReviewAspects]:
    out: list[ReviewAspects] = []
    for batch in batched(rows, batch_size):
        corpus = build_corpus(batch)
        try:
            out.extend(call(list[ReviewAspects], ASPECTS_SYSTEM, corpus))
        except Exception as e:  # noqa: BLE001
            print(f" аспекты батча {[r['review_id'] for r in batch]} не прошли: {e}")
    return out


def check_quotes(pairs: list[tuple[str, str]], source: str) -> list[tuple[str, str]]:
    ghosts: list[tuple[str, str]] = []
    for owner, quote in pairs:
        probe = quote.strip().lower()[:30]
        if probe and probe not in source:
            ghosts.append((owner, quote))
    return ghosts


def aspect_quote_pairs(aspects: list[ReviewAspects]) -> list[tuple[str, str]]:
    return [(p.review_id, a.quote) for p in aspects for a in p.aspects]


def issue_quote_pairs(reviews: list[Review]) -> list[tuple[str, str]]:
    return [(r.review_id, i.quote) for r in reviews for i in r.issues]


def build_heatmap(aspects: list[ReviewAspects], out_path: str, max_rows: int = 45) -> None:
    
    rows = [p for p in aspects if p.aspects][:max_rows]
    names = [p.review_id for p in rows]
    if not names:
        print(" heatmap пропущен: ни одной аспектной оценки")
        return
    s2n = {"positive": 1, "neutral": 0, "negative": -1}
    matrix = np.full((len(names), len(ASPECTS)), np.nan)
    for i, p in enumerate(rows):
        for a in p.aspects:
            if a.aspect in ASPECTS:
                matrix[i, ASPECTS.index(a.aspect)] = s2n[a.sentiment]
    plt.figure(figsize=(8, max(4, len(names) * 0.28)))
    sns.heatmap(
        matrix,
        annot=False,
        xticklabels=[ASPECT_RU[a] for a in ASPECTS],
        yticklabels=names,
        center=0,
        cmap="RdYlGn",
        cbar_kws={"label": "тональность (-1 нег / 0 нейтр / +1 поз)"},
        linewidths=0.4,
        linecolor="white",
    )
    plt.title("Аспектная тональность по отзывам (Instagram Lite)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# Раунд 3 — Map-Reduce
def split_into_chunks(rows: list[dict], per_chunk: int = 8) -> list[list[dict]]:
    return list(batched(rows, per_chunk))


def summarize_chunk(chunk: list[dict]) -> ChunkSummary:
    return call(ChunkSummary, CHUNK_SYSTEM, build_corpus(chunk))


def reduce_summaries(summaries: list[ChunkSummary], strict: bool = False) -> AppSummary:
    joined = "\n\n".join(
        f"## Пачка {', '.join(s.covers)} ({s.sentiment})\n"
        + "\n".join(f"- {p}" for p in s.key_points)
        for s in summaries
    )
    system = REDUCE_SYSTEM_STRICT if strict else REDUCE_SYSTEM
    return call(AppSummary, system, joined)


def summarize_app(rows: list[dict], workers: int = 6) -> list[ChunkSummary]:
    chunks = split_into_chunks(rows)
    n = len(chunks)
    print(f"   [MR] MAP: {n} пачек, до {workers} параллельно...")
    summaries: list[ChunkSummary | None] = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(summarize_chunk, c): i for i, c in enumerate(chunks)}
        for fut in as_completed(futs):
            summaries[futs[fut]] = fut.result()
    return [s for s in summaries if s is not None]


# Раунд 5 — Судья
def build_evidence_packet(reviews: list[Review], summary: AppSummary) -> str:
    parts = ["## Рекомендации (которые оцениваем)"]
    for i, a in enumerate(summary.action_items, 1):
        parts.append(f"  {i}. {a}")
    parts.append("\n## Проблемы из отзывов (исходные данные)")
    for r in reviews:
        for c in r.issues:
            parts.append(
                f"  - [{r.review_id}/{c.category}, sev={c.severity}] «{c.quote}»"
            )
    return "\n".join(parts)


def judge(reviews: list[Review], summary: AppSummary) -> JudgeReport:
    return call(JudgeReport, JUDGE_SYSTEM, build_evidence_packet(reviews, summary))

# Раунд 2.5 — Autodiscovery (на «отлично»)
def discover_aspects(rows: list[dict]) -> DiscoveredAspects:
    sample = build_corpus(rows[:24])
    return call(DiscoveredAspects, DISCOVER_SYSTEM, sample)


def classify_with_discovered(
    rows: list[dict], discovered: DiscoveredAspects, batch_size: int = 6
) -> list[DynamicReview]:
    block = "\n".join(f"- {a.name}: {a.description}" for a in discovered.aspects)
    sys_prompt = ASPECTS_SYSTEM + "\n\nИспользуй строго эти аспекты:\n" + block
    out: list[DynamicReview] = []
    for batch in batched(rows, batch_size):
        try:
            out.extend(call(list[DynamicReview], sys_prompt, build_corpus(batch)))
        except Exception as e:  # noqa: BLE001
            print(f"   dyn-аспекты батча не прошли: {e}")
    return out

def write_findings(out: Path, ctx: dict) -> None:
    g = ctx
    ghost_lines = "\n".join(
        f"- `{rid}`: «{q[:90]}»" for rid, q in g["ghost_examples"]
    ) or "- (ghost-цитат не найдено — модель цитировала дословно)"
    weak = g["weak_example"]
    weak_block = (
        f"- Рекомендация: «{weak['action']}»\n"
        f"  - вердикт судьи: `{weak['support']}`\n"
        f"  - комментарий: {weak['comment']}"
        if weak
        else "- (судья не пометил ни одной рекомендации как слабо/необоснованную)"
    )
    disc_new = ", ".join(g["discovered_new"]) or "(новых тем вне Literal не появилось)"
    disc_missing = ", ".join(g["literal_missing"]) or "(все Literal-аспекты обсуждались)"


# Оркестратор
def analyze(input_path: str, out_dir: str = "output") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_reviews(input_path)
    source = raw_text(rows)
    t0 = time.time()

    # Раунд 1: IE
    print("Раунд 1: IE (извлечение отзывов и проблем)...")
    reviews, ie_stats = extract_reviews(rows)
    (out / "reviews.json").write_text(
        json.dumps([r.model_dump() for r in reviews], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "review_id": r.review_id,
                "author": r.author,
                "rating": r.rating,
                "date": r.date,
                "category": i.category,
                "severity": i.severity,
                "quote": i.quote,
            }
            for r in reviews
            for i in (r.issues or [None])
            if i is not None
        ]
    ).to_csv(out / "reviews.csv", index=False, encoding="utf-8")
    n_issues = sum(len(r.issues) for r in reviews)
    print(f"   валидных отзывов {ie_stats['valid']}/{ie_stats['expected']}, проблем {n_issues}")

    # Раунд 2: аспекты
    print("Раунд 2: аспектный анализ + heatmap...")
    aspects = extract_aspects(rows)
    (out / "aspects.json").write_text(
        json.dumps([p.model_dump() for p in aspects], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"review_id": p.review_id, "aspect": a.aspect, "sentiment": a.sentiment,
             "confidence": a.confidence, "quote": a.quote}
            for p in aspects for a in p.aspects
        ]
    ).to_csv(out / "aspects.csv", index=False, encoding="utf-8")
    build_heatmap(aspects, str(out / "heatmap.png"))

    all_pairs = issue_quote_pairs(reviews) + aspect_quote_pairs(aspects)
    ghosts = check_quotes(all_pairs, source)
    quotes_total = len(all_pairs)
    ghost_pct = 100 * len(ghosts) / quotes_total if quotes_total else 0.0
    print(f"   ghost-цитат: {len(ghosts)}/{quotes_total} ({ghost_pct:.1f}%)")

    print("Раунд 2.5: autodiscovery аспектов")
    discovered = discover_aspects(rows)
    dyn = classify_with_discovered(rows, discovered)
    (out / "aspects_discovered.json").write_text(
        json.dumps([p.model_dump() for p in dyn], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fixed_set = set(ASPECTS)
    discovered_names = {a.name for a in discovered.aspects}
    used_dyn = {a.aspect for p in dyn for a in p.aspects}
    compare = {
        "fixed_literal": sorted(fixed_set),
        "discovered": sorted(discovered_names),
        "discovered_not_in_literal": sorted(discovered_names - fixed_set),
        "literal_not_really_discussed": sorted(fixed_set - used_dyn),
    }
    (out / "aspect_discovery_compare.json").write_text(
        json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Раунд 3: Map-Reduce
    print("Раунд 3: Map-Reduce-резюме...")
    chunk_summaries = summarize_app(rows)
    summary = reduce_summaries(chunk_summaries, strict=False)

    # Раунд 5: судья
    print("Раунд 5: LLM-as-judge...")
    report = judge(reviews, summary)
    rerun = False
    if report.overall_score < 0.7:
        print(f"   overall_score={report.overall_score:.2f} < 0.7 → строгий REDUCE...")
        rerun = True
        summary = reduce_summaries(chunk_summaries, strict=True)
        report = judge(reviews, summary)
        print(f"   после перепрогона: {report.overall_score:.2f}")
    (out / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    (out / "judge_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")

    fidelity = (
        sum(1 for _, q in all_pairs if q.strip().lower()[:30] in source) / quotes_total
        if quotes_total else 0.0
    )
    elapsed = time.time() - t0
    cost = usage_cost()
    metrics = {
        "n_input": len(rows),
        "ie": ie_stats,
        "n_issues": n_issues,
        "n_aspect_marks": quotes_total - n_issues,
        "ghosts_total": len(ghosts),
        "quotes_total": quotes_total,
        "ghost_pct": round(ghost_pct, 2),
        "fidelity": round(fidelity, 4),
        "overall_score": report.overall_score,
        "reduce_reran_strict": rerun,
        "elapsed_sec": round(elapsed, 1),
        "usage": cost,
        "aspect_discovery": compare,
        "verdict_counts": dict(Counter(v.support for v in report.verdicts)),
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    weak = next(
        (v for v in report.verdicts if v.support in ("weakly_supported", "not_supported")),
        None,
    )
    write_findings(out, {
        "n_input": len(rows), "ie": ie_stats, "n_issues": n_issues,
        "n_aspect_marks": quotes_total - n_issues,
        "ghosts_total": len(ghosts), "quotes_total": quotes_total, "ghost_pct": ghost_pct,
        "ghost_examples": ghosts[:5], "fidelity": fidelity,
        "score": report.overall_score, "rerun": rerun, "elapsed": elapsed, "cost": cost,
        "discovered_new": compare["discovered_not_in_literal"],
        "literal_missing": compare["literal_not_really_discussed"],
        "weak_example": ({"action": weak.action, "support": weak.support,
                          "comment": weak.comment} if weak else None),
    })

    print("ИТОГ")
    print(summary.headline)
    print("\nКлючевые выводы:")
    for kf in summary.key_findings:
        print(f"  • {kf}")
    print("\nРекомендации:")
    for ai in summary.action_items:
        print(f"  → {ai}")
    print(f"\noverall_score судьи: {report.overall_score:.2f}")
    print(f"ghost-цитат: {len(ghosts)}/{quotes_total} ({ghost_pct:.1f}%)")
    print(f"время: {elapsed:.1f}с, стоимость ≈ ${cost['cost_usd']}")
    print(f"\nВсе артефакты в: {out}/")


def main() -> None:
    if len(sys.argv) < 2:
        print("Использование: python pipeline.py <input/reviews.csv> [out_dir]")
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output")


if __name__ == "__main__":
    main()
