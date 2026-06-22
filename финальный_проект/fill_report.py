

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "output" / "eval_results.json"
TRACES = HERE / "output" / "traces.jsonl"
REPORT = HERE / "отчёт.md"
FIT_LABELS = ("No Fit", "Potential Fit", "Good Fit")


def _confusion_md(s: dict) -> str:
    conf = s["confusion_matrix"]
    labels = s.get("labels") or list(conf.keys())
    lines = ["| gold \\ pred | " + " | ".join(labels) + " |",
             "|" + "---|" * (len(labels) + 1)]
    for g in labels:
        lines.append("| " + g + " | " + " | ".join(str(conf[g][p]) for p in labels) + " |")
    out = "\n".join(lines)
    if s.get("fit_metrics"):
        fm = s["fit_metrics"]
        out += (f"\n\nКласс **Fit**: precision {fm['precision']}, "
                f"recall {fm['recall']}, F1 {fm['f1']}. Режим: бинарный скрининг.")
    return out


def _failure_examples() -> str:
    if not TRACES.exists():
        return "_(нет output/traces.jsonl — запустите eval.py)_"
    rows = [json.loads(l) for l in TRACES.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    # 1-2 misclassifications
    miss = [r for r in rows if r.get("gold") != r.get("pred")][:2]
    for r in miss:
        out.append(
            f"- **{r['id']}**: gold `{r['gold']}` → предсказано `{r['pred']}`. "
            f"matched={r.get('matched_skills')}, missing={r.get('missing_skills')}. "
            f"Вероятная причина — лексический промах ретрива/разногласие разметки."
        )
    # 1 ghost example
    ghosty = [r for r in rows if (r.get("ghost") or {}).get("ghost_examples")][:1]
    for r in ghosty:
        ex = (r["ghost"]["ghost_examples"] or ["—"])[0]
        out.append(
            f"- **{r['id']}**: модель привела цитату-призрак (нет в резюме): "
            f"«{ex}…» — детектор отбросил её и понизил вердикт."
        )
    return "\n".join(out) or "_(в этом прогоне явных провалов не нашлось)_"


def main() -> None:
    if not RES.exists():
        raise SystemExit("Нет output/eval_results.json — сначала: python eval.py")
    s = json.loads(RES.read_text(encoding="utf-8"))
    h = s.get("hallucination", {})
    j = s.get("judge_avg", {})
    p = s.get("path", {})
    repl = {
        "{{N}}": (f"{s.get('n')} (успешно оценено {s.get('scored', s.get('n'))}, "
                  f"ошибок API {s.get('errors', 0)})"),
        "{{exact_accuracy}}": s.get("exact_accuracy"),
        "{{adjacent_accuracy}}": s.get("adjacent_accuracy"),
        "{{macro_f1}}": s.get("macro_f1"),
        "{{judge_grounded}}": j.get("groundedness"),
        "{{judge_relevance}}": j.get("relevance"),
        "{{avg_steps}}": p.get("avg_steps"),
        "{{avg_tools}}": p.get("avg_tool_calls"),
        "{{llm_calls}}": p.get("llm_calls"),
        "{{tokens}}": p.get("total_tokens"),
        "{{cost}}": p.get("approx_cost_usd"),
        "{{ghost_share}}": h.get("share_with_ghost"),
        "{{ghost_quotes}}": h.get("ghost_quotes"),
        "{{total_evidence}}": h.get("total_evidence"),
        "{{ghost_quote_rate}}": h.get("ghost_quote_rate"),
        "{{ghost_skills}}": h.get("ghost_skills"),
        "{{CONFUSION_TABLE}}": _confusion_md(s),
        "{{FAILURE_EXAMPLES}}": _failure_examples(),
    }
    text = REPORT.read_text(encoding="utf-8")
    for k, v in repl.items():
        text = text.replace(k, str(v))
    REPORT.write_text(text, encoding="utf-8")
    print(f"Отчёт обновлён числами: {REPORT}")


if __name__ == "__main__":
    main()
