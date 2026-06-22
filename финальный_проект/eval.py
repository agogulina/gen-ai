

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_client
from hallucination import aggregate_ghost
from judge import judge_assessment
from pipeline import assess_pair
from schema import FIT_LABELS, FIT_ORDER

HERE = Path(__file__).resolve().parent
EVAL_SET = HERE / "input" / "eval_set.jsonl"
OUT = HERE / "output"

PRICE_IN = 0.27 / 1_000_000
PRICE_OUT = 1.10 / 1_000_000


def _load_eval() -> list[dict]:
    if not EVAL_SET.exists():
        sys.exit("Нет input/eval_set.jsonl — сначала: python prepare_data.py")
    with open(EVAL_SET, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _macro_f1(conf: dict[str, dict[str, int]], labels) -> float:
    f1s = []
    for lab in labels:
        tp = conf[lab][lab]
        fp = sum(conf[o][lab] for o in labels if o != lab)
        fn = sum(conf[lab][o] for o in labels if o != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s)


def _to_binary(label: str) -> str:
    return "No Fit" if label == "No Fit" else "Fit"


def _prf_positive(conf: dict, pos: str = "Fit") -> dict:
    tp = conf[pos][pos]
    fp = sum(conf[o][pos] for o in conf if o != pos)
    fn = sum(conf[pos][o] for o in conf[pos] if o != pos)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def run(limit: int | None, use_skeptic: bool, use_judge: bool) -> dict:
    OUT.mkdir(exist_ok=True)
    data = _load_eval()
    if limit:
        data = data[:limit]
    llm_client.reset_usage()
    t0 = time.perf_counter()

    gold_set = {item["label"] for item in data}
    BINARY = gold_set <= {"No Fit", "Fit"}
    LABELS = ("No Fit", "Fit") if BINARY else FIT_LABELS
    ORDER = {"No Fit": 0, "Fit": 1} if BINARY else FIT_ORDER

    conf = {g: {p: 0 for p in LABELS} for g in LABELS}
    correct = adjacent = scored = 0
    n_errors = 0
    ghost_objs = []
    judges = []
    steps_sum = tools_sum = 0
    preds = []
    traces_f = open(OUT / "traces.jsonl", "w", encoding="utf-8")

    for i, item in enumerate(data, 1):
        gold = item["label"]
        errored = False
        try:
            res = assess_pair(item["resume_text"], item["job_description_text"],
                              use_skeptic=use_skeptic)
        except Exception as exc:
            print(f"  [{i}/{len(data)}] {item['id']}: ОШИБКА {exc}")
            n_errors += 1
            errored = True
            res = {"final_fit": None, "matched_skills": [], "missing_skills": [],
                   "evidence": [], "rationale": f"EXC {exc}", "ghost": {},
                   "_ghost_obj": None, "n_steps": 0, "n_tool_calls": 0,
                   "requirements": {}, "confidence": 0.0, "skeptic": None}
        pred3 = res["final_fit"]
        gold_l = _to_binary(gold) if BINARY else gold
        pred = (_to_binary(pred3) if pred3 else None) if BINARY else pred3
        if not errored and pred is not None:
            conf[gold_l][pred] += 1
            scored += 1
            correct += int(pred == gold_l)
            adjacent += int(abs(ORDER[pred] - ORDER[gold_l]) <= 1)
            if res.get("_ghost_obj") is not None:
                ghost_objs.append(res["_ghost_obj"])
            steps_sum += res.get("n_steps", 0)
            tools_sum += res.get("n_tool_calls", 0)

        jv = None
        if use_judge and not errored:
            try:
                jv = judge_assessment(res["requirements"], res)
                judges.append(jv)
            except Exception as exc:  # noqa: BLE001
                print(f"    judge error: {exc}")

        if errored:
            print(f"  [{i}/{len(data)}] {item['id']}: ПРОПУЩЕН (ошибка API) — не учтён в accuracy")
        else:
            print(f"  [{i}/{len(data)}] {item['id']}: gold={gold_l} pred={pred} "
                  f"{'OK' if pred==gold_l else '×'} ghost={res['ghost'].get('had_any_ghost')}")

        row = {
            "id": item["id"], "gold": gold_l, "pred": pred, "pred_3class": pred3,
            "confidence": res.get("confidence"),
            "matched_skills": res.get("matched_skills"),
            "missing_skills": res.get("missing_skills"),
            "n_steps": res.get("n_steps"), "n_tool_calls": res.get("n_tool_calls"),
            "ghost": res.get("ghost"),
            "skeptic_adjusted": (res.get("skeptic") or {}).get("adjusted_fit"),
            "judge_groundedness": jv.groundedness if jv else None,
            "judge_relevance": jv.relevance if jv else None,
        }
        preds.append(row)
        traces_f.write(json.dumps({**row, "rationale": res.get("rationale"),
                                   "evidence": res.get("evidence")},
                                  ensure_ascii=False) + "\n")
    traces_f.close()

    n = len(data)
    denom = scored or 1
    usage = llm_client.get_usage()
    cost = usage["prompt_tokens"] * PRICE_IN + usage["completion_tokens"] * PRICE_OUT
    ghost_agg = aggregate_ghost(ghost_objs) if ghost_objs else {}
    judge_avg = {
        "groundedness": round(sum(j.groundedness for j in judges) / len(judges), 2) if judges else None,
        "relevance": round(sum(j.relevance for j in judges) / len(judges), 2) if judges else None,
    }
    summary = {
        "n": n,
        "mode": "binary" if BINARY else "3-class",
        "labels": list(LABELS),
        "scored": scored,
        "errors": n_errors,
        "exact_accuracy": round(correct / denom, 3),
        "adjacent_accuracy": round(adjacent / denom, 3),
        "macro_f1": round(_macro_f1(conf, LABELS), 3),
        "confusion_matrix": conf,
        "judge_avg": judge_avg,
        "path": {
            "avg_steps": round(steps_sum / denom, 2),
            "avg_tool_calls": round(tools_sum / denom, 2),
            "llm_calls": usage["calls"],
            "total_tokens": usage["total_tokens"],
            "approx_cost_usd": round(cost, 4),
            "wall_sec": round(time.perf_counter() - t0, 1),
        },
        "hallucination": ghost_agg,
        "config": {"use_skeptic": use_skeptic, "use_judge": use_judge},
    }
    if BINARY:
        summary["fit_metrics"] = _prf_positive(conf, "Fit")  

    (OUT / "eval_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(OUT / "predictions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0].keys()) if preds else ["id"])
        w.writeheader()
        for r in preds:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                        for k, v in r.items()})
    _write_table(summary)
    _print_summary(summary)
    return summary


def _write_table(s: dict) -> None:
    labels = s.get("labels", list(FIT_LABELS))
    mode = s.get("mode", "3-class")
    head = f"# Eval — матчинг резюме ↔ вакансия ({mode})\n"
    L = [head,
         f"Пар всего: **{s['n']}** (успешно оценено **{s.get('scored', s['n'])}**, "
         f"ошибок API: **{s.get('errors', 0)}**) | exact acc **{s['exact_accuracy']}** | "
         f"adjacent acc **{s['adjacent_accuracy']}** | macro-F1 **{s['macro_f1']}**\n"]
    if s.get("fit_metrics"):
        fm = s["fit_metrics"]
        L.append(f"Класс **Fit**: precision **{fm['precision']}**, recall **{fm['recall']}**, "
                 f"F1 **{fm['f1']}**\n")
    L += ["## Матрица ошибок (строки — gold, столбцы — pred)\n",
          "| gold \\ pred | " + " | ".join(labels) + " |",
          "|" + "---|" * (len(labels) + 1)]
    for g in labels:
        c = s["confusion_matrix"][g]
        L.append("| " + g + " | " + " | ".join(str(c[p]) for p in labels) + " |")
    L += ["\n## Путь и стоимость\n",
          f"- среднее шагов агента: {s['path']['avg_steps']}",
          f"- среднее вызовов инструментов: {s['path']['avg_tool_calls']}",
          f"- всего вызовов LLM: {s['path']['llm_calls']}, токенов: {s['path']['total_tokens']}",
          f"- ориентировочная стоимость: ${s['path']['approx_cost_usd']}",
          f"- судья (1..5): groundedness {s['judge_avg']['groundedness']}, relevance {s['judge_avg']['relevance']}",
          "\n## Галлюцинации\n",
          f"- доля ответов с ghost: {s['hallucination'].get('share_with_ghost')}",
          f"- ghost-цитат: {s['hallucination'].get('ghost_quotes')} из {s['hallucination'].get('total_evidence')} "
          f"(rate {s['hallucination'].get('ghost_quote_rate')})",
          f"- ghost-навыков: {s['hallucination'].get('ghost_skills')}"]
    (OUT / "eval_table.md").write_text("\n".join(L), encoding="utf-8")


def _print_summary(s: dict) -> None:
    print("\n" + "=" * 60)
    print(f"[{s.get('mode')}] N={s['n']} (оценено {s.get('scored')}, ошибок API {s.get('errors')})  "
          f"exact={s['exact_accuracy']}  adjacent={s['adjacent_accuracy']}  macroF1={s['macro_f1']}")
    if s.get("fit_metrics"):
        print("Fit class:", s["fit_metrics"])
    print("judge:", s["judge_avg"], "| path:", s["path"])
    print("hallucination:", s["hallucination"])
    print(f"Артефакты в {OUT}/ : eval_results.json, eval_table.md, predictions.csv, traces.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-skeptic", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    run(args.limit, use_skeptic=not args.no_skeptic, use_judge=not args.no_judge)
