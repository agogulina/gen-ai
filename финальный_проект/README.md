# Матчинг резюме - вакансия с проверяемым обоснованием (Трек B)

Система оценивает соответствие кандидата вакансии (`Fit` / `No Fit`, в 3-классовом
режиме — `No Fit` / `Potential Fit` / `Good Fit`) и выдаёт структурированное
обоснование с дословными цитатами из резюме, проверяя их на галлюцинации.
Артефакты в `output/`.

## Установка
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         
```

## Запуск

**Основной прогон — синтетический датасет** (в `input/eval_set.jsonl`):
```bash
python eval.py           
```
`output/`: `eval_results.json`, `eval_table.md`, `predictions.csv`, `traces.jsonl`.

**Дополнительный прогон — публичный датасет (3 класса).**
Положите `train.csv`/`test.csv` (HF `cnamuangtoun/resume-job-description-fit`) в `input/`, затем:
```bash
python prepare_data.py --per-class 6   # (18 пар)
python eval.py
```

Результаты 3-классового прогона приложены отдельно в папке `результаты_прошлые/`.

## Что где лежит
| Файл | Роль |
|---|---|
| `schema.py` | pydantic-модели + `field_validator`/`model_validator`  |
| `retrieval.py` | RAG: Okapi BM25 по фрагментам резюме  |
| `tools.py` | инструменты агента: `search_resume`, `check_skill_present` |
| `agent.py` | извлечение требований + ReAct-агент + покрытие навыков + `FitAssessment` |
| `critic.py` | скептик (второй агент, перепроверка на завышение) |
| `hallucination.py` | детектор ghost-цитат/навыков + понижение вердикта + метрики |
| `pipeline.py` | сквозной конвейер одной пары |
| `judge.py` | LLM-as-judge (groundedness/relevance) |
| `eval.py` | прогон на ≥15 входах: правильность + путь + галлюцинации |
| `prepare_data.py` | выборка из публичного `test.csv` (для доп. прогона) |
| `llm_client.py` | DeepSeek-клиент (JSON-инструктор + raw) + учёт токенов |
| `input/eval_set.jsonl` | синтетический датасет (основной прогон) |
| `output/` | артефакты прогона |
| `результаты_прошлые/` | артефакты 3-классового прогона на публичных данных |

## Техники курса
RAG (BM25), агент с инструментами (ReAct), мультиагент (матчер + скептик),
структурированный вывод (`response_model` + `max_retries` + валидаторы),
LLM-as-judge, проверка галлюцинаций. Подробности — в `отчёт.md`.
