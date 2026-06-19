
set -e
cd "$(dirname "$0")"

if [ ! -f .env ] && [ -z "$LLM_AUTH_TOKEN" ]; then
  echo "Нет .env и не задан LLM_AUTH_TOKEN."
  echo "Сделайте: cp .env.example .env  и впишите ключ DeepSeek (LLM_AUTH_TOKEN)."
  exit 1
fi

: "${CRITIC_N:=10}"
: "${EVAL_N:=5}"
export CRITIC_N EVAL_N

echo "==> [1/5] Демо валидатора схемы (без сети)"
python3 demo_validator.py

echo "==> [2/5] Замер параллельности (Q1 и Q5)"
python3 measure_parallel.py

echo "==> [3/5] Замер критики T=0.0 vs T=0.7, N=$CRITIC_N (~$((5*2*CRITIC_N)) запросов)"
python3 measure_critic.py

echo "==> [4/5] Eval 6 вопросов x 3 конфигурации, N=$EVAL_N (~$((6*3*EVAL_N)) запросов)"
python3 eval_pwc6.py -n "$EVAL_N"

echo "==> [5/5] Сборка отчёта"
python3 generate_report.py

echo
echo "Готово. Отчёт: otchet.md   Сырые JSON: validator_demo.json parallel.json critic.json eval6.json"
