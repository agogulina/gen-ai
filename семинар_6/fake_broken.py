
from __future__ import annotations

from schemas_pwc import Plan, SubQuestion, WorkerAnswer


def _wa(sid, q, ans, tools):
    return WorkerAnswer(
        subquestion_id=sid, question_snippet=q[:60], answer=ans, used_tools=tools
    )


FAKE_BROKEN: list[dict] = [
    # арифметика без calculate
    {
        "label": "арифметика без calculate",
        "question": "На сколько рублей курс USD в декабре 2024 выше, чем в январе 2022?",
        "plan": Plan(
            reasoning="Взять два курса и вычесть.",
            subquestions=[
                SubQuestion(id=1, question="курс USD на 2022-01-15?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="курс USD на 2024-12-15?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=3, question="разница курсов?", expected_tools=["calculate"], depends_on=[1, 2]),
            ],
        ),
        "answers": {
            1: _wa(1, "курс USD на 2022-01-15?", "USD = 76.2 руб.", ["get_fx_rate"]),
            2: _wa(2, "курс USD на 2024-12-15?", "USD = 103.4 руб.", ["get_fx_rate"]),
            # разница без calculate и неверная (27.2 вместо 27.2 -> намеренно 25):
            3: _wa(3, "разница курсов?", "Разница примерно 25 рублей.", []),
        },
    },
    # выдуманное число (значения нет ни в одном инструменте)
    {
        "label": "выдуманное число",
        "question": "Какая была ключевая ставка ЦБ в марте 2022?",
        "plan": Plan(
            reasoning="Ставка ЦБ на дату.",
            subquestions=[
                SubQuestion(id=1, question="ключевая ставка на 2022-03-15?", expected_tools=["get_key_rate"]),
            ],
        ),
        "answers": {
            1: _wa(1, "ключевая ставка на 2022-03-15?", "Ставка ЦБ — 9.5% годовых.", ["get_key_rate"]),
        },
    },
    # несогласованные данные между подвопросами (один месяц — два разных курса)
    {
        "label": "несогласованные данные",
        "question": "Каким был курс доллара на 2024-12-15?",
        "plan": Plan(
            reasoning="Курс USD нужен дважды для двух веток расчёта.",
            subquestions=[
                SubQuestion(id=1, question="курс USD на 2024-12-15 (ветка A)?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="курс USD на 2024-12-15 (ветка B)?", expected_tools=["get_fx_rate"]),
            ],
        ),
        "answers": {
            1: _wa(1, "курс USD на 2024-12-15 (ветка A)?", "USD = 103.4 руб.", ["get_fx_rate"]),
            2: _wa(2, "курс USD на 2024-12-15 (ветка B)?", "USD = 88.0 руб.", ["get_fx_rate"]),  # противоречие
        },
    },
    # 4) неправильный инструмент: инфляцию "взяли" из get_key_rate
    {
        "label": "неверный инструмент",
        "question": "Какая инфляция (ИПЦ г/г) была в январе 2024?",
        "plan": Plan(
            reasoning="ИПЦ за январь 2024.",
            subquestions=[
                SubQuestion(id=1, question="инфляция за 2024-01?", expected_tools=["get_inflation"]),
            ],
        ),
        "answers": {
            1: _wa(1, "инфляция за 2024-01?", "Инфляция 16% (по данным о ставке).", ["get_key_rate"]),
        },
    },
    # 5) не отвечает на вопрос 
    {
        "label": "ответ не по вопросу",
        "question": "Во сколько раз вырос курс доллара с января 2022 по декабрь 2024?",
        "plan": Plan(
            reasoning="Отношение двух курсов.",
            subquestions=[
                SubQuestion(id=1, question="курс USD на 2022-01-15?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="курс USD на 2024-12-15?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=3, question="во сколько раз вырос?", expected_tools=["calculate"], depends_on=[1, 2]),
            ],
        ),
        "answers": {
            1: _wa(1, "курс USD на 2022-01-15?", "USD = 76.2 руб.", ["get_fx_rate"]),
            2: _wa(2, "курс USD на 2024-12-15?", "USD = 103.4 руб.", ["get_fx_rate"]),
            3: _wa(3, "во сколько раз вырос?", "Курс в декабре 2024 равен 103.4 руб.", []),  # не отношение
        },
    },
]
