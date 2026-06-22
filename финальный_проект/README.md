# Матчинг резюме ↔ вакансия с проверяемым обоснованием (Трек B)

Система оценивает соответствие кандидата вакансии (`No Fit` / `Potential Fit` /
`Good Fit`) и выдаёт **структурированное обоснование с дословными цитатами из
резюме**, проверяя их на галлюцинации. Написано руками, без LangChain/CrewAI;
LLM-бэкенд — DeepSeek; интерфейс — CLI, артефакты в `output/`.

## Данные
Реальный датасет Hugging Face `cnamuangtoun/resume-job-description-fit`
(колонки `resume_text`, `job_description_text`, `label`; метки
`No Fit / Potential Fit / Good Fit`). Положите `train.csv` и `test.csv` в
`input/` (уже там, если разворачиваете архив целиком). Источник датасета —
сторонний; используется как есть, метки идут как gold для оценки правильности.

## Установка
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # впишите LLM_AUTH_TOKEN (ключ DeepSeek)
```

## Запуск одной командой
```bash
chmod +x run.sh && ./run.sh
```
Это: выборка eval-набора → прогон конвейера с метриками → заполнение `отчёт.md`.
Дымовой прогон (быстро, мало запросов к API):
```bash
LIMIT=3 PER_CLASS=5 ./run.sh
```

## По шагам
```bash
python prepare_data.py --per-class 6     # input/eval_set.jsonl (18 пар, >=15)
python eval.py                           # output/* + метрики
python eval.py --limit 6 --no-judge      # быстрый прогон без судьи
python pipeline.py                       # одна демо-пара (резюме+вакансия)
python fill_report.py                    # подставить числа в отчёт.md
```

## Что где лежит
| Файл | Роль |
|---|---|
| `schema.py` | pydantic-модели + `field_validator`/`model_validator` (бизнес-инварианты) |
| `retrieval.py` | RAG: Okapi BM25 по фрагментам резюме (руками) |
| `tools.py` | инструменты агента: `search_resume`, `check_skill_present` |
| `agent.py` | матчер: извлечение требований + ReAct-цикл + структурный `FitAssessment` |
| `critic.py` | скептик (второй агент, перепроверка на завышение) |
| `hallucination.py` | детектор ghost-цитат/навыков + понижение вердикта + метрики |
| `pipeline.py` | сквозной конвейер одной пары |
| `judge.py` | LLM-as-judge (groundedness/relevance) для eval |
| `eval.py` | прогон на ≥15 входах: правильность + путь + галлюцинации |
| `prepare_data.py` | стратифицированная выборка из `input/test.csv` |
| `fill_report.py` | подстановка метрик/примеров в `отчёт.md` |
| `llm_client.py` | DeepSeek-клиент (JSON-инструктор + raw) + учёт токенов |

## Техники курса (≥4)
RAG (BM25), агент с инструментами (ReAct), мультиагент (матчер + скептик),
структурированный вывод (`response_model` + `max_retries` + валидаторы),
LLM-as-judge, проверка галлюцинаций. Подробности — в `отчёт.md`.

## Бинарный режим (опционально)
Можно свести задачу к скринингу `No Fit` vs `Fit` (Potential+Good вместе) —
убирает шумную границу Potential↔Good, метрики устойчивее:
```bash
python prepare_data.py --binary        # 12+12 = 24 пары
python eval.py                         # eval сам определит бинарный режим
# или одной командой:
BINARY=1 ./run.sh
```

## Чистая выборка с явными навыками
Сигнал покрытия осмысленнее на парах, где навыки перечислены и в резюме, и в
вакансии. Рекомендуемый «чистый» прогон:
```bash
EXPLICIT=1 BINARY=1 ./run.sh        # явные навыки + бинарный скрининг
```
