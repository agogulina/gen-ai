# Как запустить (Семинар 5)

## 0. Установка (один раз)

```bash
cd seminar5_agent
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # затем впишите ключ DeepSeek в LLM_AUTH_TOKEN
```

В `.env` модель уже выставлена на `deepseek-chat` (рабочая публичная модель
DeepSeek). `LLM_AUTH_TOKEN` — ваш ключ.

## 1. Новый инструмент — контрольный вопрос из задания

```bash
python agent.py "Во сколько раз вырос курс USD с января 2022 по апрель 2026?"
```
Агент должен один раз позвать `compare_periods` и ответить (ratio ≈ 1.08).
Проверить инструмент без агента/LLM можно офлайн:
```bash
python -c "from tools import compare_periods; print(compare_periods('fx_USD','2022-01','2026-04'))"
```

## 2. Лог шагов в JSONL

Любой запуск `agent.py`/`eval.py` дописывает события в `trace.jsonl`
(режим append, `run_id` = uuid4 на прогон). Посмотреть последние строки:
```bash
tail -n 20 trace.jsonl
```

## 3. Eval на 10 вопросов (нужен ключ)

```bash
python eval.py
```
Печатает прогон по каждому вопросу и итог «прошло N/10». Создаёт:
- `eval_results.json` — детальные результаты,
- `eval_table.md` — готовая таблица id/query/ok?/steps/tools_used для отчёта.

Содержимое `eval_table.md` вставьте в раздел 2 отчёта `report.md`.

## 4. Диагностика ошибок (офлайн, без ключа)

```bash
python diagnose.py
```
Детерминированно воспроизводит 3 разных типа ошибок (битый JSON / галлюцинация
инструмента / падение на плохих аргументах) + бонусный 4-й (нет данных), дописывая
их в `trace.jsonl`. Разбор — в разделе 3 отчёта.

## Состав

```
seminar5_agent/
├── agent.py        # ReAct-агент + JSONL-лог (ДЗ-2) + 6-й инструмент подключён
├── tools.py        # 5 инструментов + compare_periods (ДЗ-1)
├── schemas.py      # JSON-схемы, включая compare_periods
├── eval.py         # 10 вопросов (ДЗ-3), пишет eval_results.json + eval_table.md
├── diagnose.py     # харнесс диагностики ошибок (ДЗ-4)
├── report.md       # отчёт (ДЗ-5)
├── data/           # CSV ЦБ/Росстата (fallback)
├── requirements.txt
└── .env.example
```
