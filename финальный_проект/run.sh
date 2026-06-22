#!/usr/bin/env bash
# Полный прогон проекта: выборка -> eval -> заполнение отчёта.
# Дымовой прогон: LIMIT=3 PER_CLASS=5 ./run.sh
set -e
cd "$(dirname "$0")"
if [ ! -f .env ] && [ -z "$LLM_AUTH_TOKEN" ]; then
  echo "Нет .env и LLM_AUTH_TOKEN. Сделайте: cp .env.example .env и впишите ключ DeepSeek."; exit 1
fi
: "${PER_CLASS:=6}"
PREP_FLAGS=""
if [ "${BINARY:-0}" = "1" ]; then PREP_FLAGS="$PREP_FLAGS --binary"; fi
if [ "${EXPLICIT:-0}" = "1" ]; then PREP_FLAGS="$PREP_FLAGS --explicit-skills"; fi
echo "==> [1/3] Выборка eval-набора (стратиф.)"
python3 prepare_data.py --per-class "$PER_CLASS" $PREP_FLAGS
echo "==> [2/3] Eval (конвейер + судья + метрики)"
if [ -n "$LIMIT" ]; then python3 eval.py --limit "$LIMIT"; else python3 eval.py; fi
echo "==> [3/3] Заполнение отчёта числами из output/"
python3 fill_report.py
echo "Готово: output/ (eval_results.json, eval_table.md, predictions.csv, traces.jsonl), отчёт.md"
