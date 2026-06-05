# Входные данные — отзывы Google Play

**Приложение:** Instagram Lite 

## Источник (открытые данные)

Отзывы взяты из публичного репозитория с примером скрейпинга Google Play:

```
https://raw.githubusercontent.com/jtlawren67/jlawblog/master/content/post/
2021-05-03-scraping-google-play-reviews-with-rselenium/data/review_data.csv
```

Исходный файл: `_raw_instagram_lite_reviews.csv` (2040 отзывов, 2019–2021).

## Как получен `reviews.csv`

Скриптом `prepare_input.py` (в корне решения) детерминированно отобрано
**60 содержательных отзывов**, сбалансированных по оценке (по 12 на каждую
из оценок 1–5): отбрасываются слишком короткие/длинные и почти-дубликаты,
внутри оценки сортировка по числу лайков и дате. Никакие тексты не
выдумываются и не редактируются (только схлопывание пробелов).

Колонки `reviews.csv`: `review_id, author, rating, date, thumbs_up, text`.

Чтобы пересобрать вход:
```bash
python prepare_input.py
```
