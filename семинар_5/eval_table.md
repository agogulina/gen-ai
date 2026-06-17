| id | query | ok? | steps | tools_used |
|---|---|:---:|:---:|---|
| 1 | Какая сегодня ключевая ставка ЦБ? | ✅ | 2 | get_key_rate |
| 2 | Сколько стоит доллар сегодня и сколько стоил 1 января 2022? | ✅ | 2 | get_fx_rate, get_fx_rate |
| 3 | Какая сейчас реальная ключевая ставка? (номинальная минус инфляция г/г) | ✅ | 5 | get_key_rate, get_inflation, get_inflation, get_inflation, calculate |
| 4 | Посчитай, за сколько лет удвоится вклад 100 тыс руб при текущей ключевой ставке (формула 72). | ✅ | 3 | get_key_rate, calculate |
| 5 | Во сколько раз вырос курс USD с января 2022 по апрель 2026? | ✅ | 2 | compare_periods |
| 6 | На сколько процентных пунктов изменилась ключевая ставка с декабря 2021 по март 2026? | ✅ | 2 | compare_periods |
| 7 | Сравни инфляцию в марте и в третьем квартале 2024 года. | ✅ | 3 | get_inflation, get_inflation, get_inflation, get_inflation, compare_periods |
| 8 | Что было выше в апреле 2026: ключевая ставка или годовая инфляция? | ✅ | 3 | get_key_rate, get_inflation, get_inflation |
| 9 | Какой сейчас «индекс нищеты» (инфляция г/г плюс безработица)? | ✅ | 5 | get_inflation, get_unemployment, get_inflation, get_unemployment, get_inflation, calculate |
| 10 | Какова реальная доходность годового рублёвого вклада при текущей ставке и инфляции? | ✅ | 5 | get_key_rate, get_inflation, get_inflation, get_inflation, calculate |

**Прошло: 10/10**
