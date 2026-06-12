

from __future__ import annotations

import json
import os
from pathlib import Path

from pipeline import BACKEND, Index

DATA_DIR = Path(__file__).parent / "data"
STRATEGIES = ["fixed", "recursive"]
K = 5


def load_gold() -> list[dict]:
    return json.loads((DATA_DIR / "gold.json").read_text(encoding="utf-8"))


HARD_TYPES = {"синоним", "multi-hop"}


def is_hard(q: dict) -> bool:
    return q["type"] in HARD_TYPES


def sources_of(ids: list[str]) -> list[str]:
    seen: list[str] = []
    for cid in ids:
        src = cid.split("__")[0]
        if src not in seen:
            seen.append(src)
    return seen


def first_gold_rank(retrieved_src: list[str], gold_src: list[str]) -> int | None:
    for rank, s in enumerate(retrieved_src, start=1):
        if s in gold_src:
            return rank
    return None


def evaluate(strategy: str, gold: list[dict]) -> dict:
    idx = Index(strategy)
    rows = []
    hits = 0          # hit@5: хотя бы один gold-источник в top-5
    hits1 = 0         # hit@1: gold-источник на 1-й позиции
    recall_sum = 0.0  # средняя доля найденных gold-источников
    full_sum = 0      # full-hit@5: найдены все gold-источники
    mrr_sum = 0.0     # mean reciprocal rank первого gold-источника
    for q in gold:
        res = idx.retrieve(q["question"], k=K)
        retrieved_ids = res["ids"][0]
        retrieved_src = sources_of(retrieved_ids)
        gold_src = q["gold_sources"]
        found = [s for s in gold_src if s in retrieved_src]
        is_hit = len(found) > 0
        is_full = len(found) == len(gold_src)
        recall = len(found) / len(gold_src)
        rank1 = first_gold_rank(retrieved_src, gold_src)
        hits += int(is_hit)
        hits1 += int(rank1 == 1)
        full_sum += int(is_full)
        recall_sum += recall
        mrr_sum += (1.0 / rank1) if rank1 else 0.0
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "hard": is_hard(q),
                "gold_sources": gold_src,
                "retrieved_sources_top5": retrieved_src,
                "hit": is_hit,
                "full_hit": is_full,
                "first_gold_rank": rank1,
                "recall_at_5": round(recall, 3),
            }
        )
    n = len(gold)
    return {
        "strategy": strategy,
        "backend": idx.dense.name,
        "stats": idx.stats(),
        "n_questions": n,
        "hit_rate_at_5": round(hits / n, 3),
        "hit_rate_at_1": round(hits1 / n, 3),
        "full_hit_rate_at_5": round(full_sum / n, 3),
        "mean_recall_at_5": round(recall_sum / n, 3),
        "mrr": round(mrr_sum / n, 3),
        "rows": rows,
    }


def print_report(results: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print(f"СРАВНЕНИЕ СТРАТЕГИЙ ЧАНКИНГА | бэкенд ретрива: {BACKEND}")
    print("=" * 78)
    for strat in STRATEGIES:
        print(results[strat]["stats"])
    print("-" * 78)

    # Сводная таблица
    print(f"\n{'Стратегия':<12}{'hit@5':>9}{'hit@1':>9}{'full@5':>9}{'recall@5':>11}{'MRR':>8}")
    print("-" * 58)
    for strat in STRATEGIES:
        r = results[strat]
        print(
            f"{strat:<12}{r['hit_rate_at_5']:>9.3f}{r['hit_rate_at_1']:>9.3f}"
            f"{r['full_hit_rate_at_5']:>9.3f}{r['mean_recall_at_5']:>11.3f}{r['mrr']:>8.3f}"
        )

    print("\nПовопросно (1 = gold-источник попал в top-5, частичный recall в скобках):")
    print("-" * 78)
    hdr = f"{'id':>3} {'тип':<16}{'hard':<6}{'fixed':<16}{'recursive':<16}"
    print(hdr)
    print("-" * 78)
    gold = load_gold()
    by_id = {strat: {row["id"]: row for row in results[strat]["rows"]} for strat in STRATEGIES}
    for q in gold:
        qid = q["id"]
        cells = []
        for strat in STRATEGIES:
            row = by_id[strat][qid]
            mark = "✓" if row["hit"] else "✗"
            rk = row["first_gold_rank"]
            cells.append(f"{mark} r={rk} ({row['recall_at_5']:.2f})")
        hard = "да" if is_hard(q) else ""
        print(f"{qid:>3} {q['type']:<16}{hard:<6}{cells[0]:<20}{cells[1]:<20}")
    print("-" * 78)


def main() -> None:
    gold = load_gold()
    results = {strat: evaluate(strat, gold) for strat in STRATEGIES}
    print_report(results)
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nПодробные результаты сохранены в {out.name}")


if __name__ == "__main__":
    main()
