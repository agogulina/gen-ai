import csv
import random
import time

from llm_client import get_model, make_client
from schema import Application, CITIES

MODEL = get_model()
client = make_client()

N_TOTAL = 50

SPECIALITIES = [
    "Врач",
    "Медицинская сестра",
    "Учитель",
    "Воспитатель",
    "Социальный работник",
    "Психолог",
    "Бухгалтер",
    "Юрист",
    "Инженер",
    "Программист",
]


def _build_quota_queue() -> list[tuple[str, str]]:
    """
    Строим 50 пар (город, специальность):
    - По каждой специальности — ровно 5 заявок (10 * 5 = 50).
    - Города циклически распределяются между специальностями,
      чтобы каждый из 12 городов получил ~4 заявки.
    """
    queue: list[tuple[str, str]] = []
    city_idx = 0
    for spec in SPECIALITIES:
        for _ in range(N_TOTAL // len(SPECIALITIES)):
            queue.append((CITIES[city_idx % len(CITIES)], spec))
            city_idx += 1
    random.shuffle(queue)
    return queue


SYSTEM_PROMPT = """\
Ты — генератор тестовых данных. Создавай реалистичные анкеты-заявки на курсы \
повышения квалификации (ДПО) для российских специалистов.
Имена должны быть разнообразными: русские, татарские, башкирские, украинские, \
армянские, грузинские и т.д.
Возраст варьируй от 22 до 65 лет. Район (district) — реальный район или округ \
указанного города. Желаемый курс выбирай так, чтобы он логично сочетался со \
специальностью человека.
"""


def _make_user_prompt(seed_city: str, seed_speciality: str) -> str:
    return (
        f"Сгенерируй одну заявку на курс ДПО. "
        f"Город: {seed_city}. "
        f"Специальность заявителя: {seed_speciality}. "
        f"Подбери уместный для этой специальности курс из доступных вариантов. "
        f"Год окончания вуза и возраст должны быть согласованы (не мог окончить вуз раньше 18 лет). "
        f"Стаж работы должен соответствовать возрасту и году выпуска."
    )


def generate_application(seed_city: str, seed_speciality: str) -> Application:
    return client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _make_user_prompt(seed_city, seed_speciality)},
        ],
        response_model=Application,
        max_retries=3,
        temperature=0.9,
    )


def main():
    queue = _build_quota_queue()
    applications: list[Application] = []
    errors = 0

    print(
        f"Генерируем {N_TOTAL} заявок (стратификация: "
        f"{len(CITIES)} городов × {len(SPECIALITIES)} специальностей)...\n"
    )

    for i, (seed_city, seed_speciality) in enumerate(queue, start=1):
        print(
            f"[{i:>2}/{N_TOTAL}] {seed_city} + {seed_speciality}...",
            end=" ",
            flush=True,
        )
        try:
            app = generate_application(seed_city, seed_speciality)
            applications.append(app)
            print(
                f"✓  {app.full_name}, {app.age} лет, "
                f"→ {app.desired_course[:30]}..."
            )
        except Exception as e:
            errors += 1
            print(f"✗  Ошибка: {type(e).__name__}: {e}")
        time.sleep(0.3)

    print(f"\nИтого: {len(applications)} успешных, {errors} ошибок.")

    fieldnames = [
        "full_name", "age", "city", "district",
        "speciality", "desired_course",
        "years_of_experience", "graduation_year",
    ]
    with open("applications.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for app in applications:
            writer.writerow({
                "full_name": app.full_name,
                "age": app.age,
                "city": app.address.city,
                "district": app.address.district,
                "speciality": app.speciality,
                "desired_course": app.desired_course,
                "years_of_experience": app.years_of_experience,
                "graduation_year": app.graduation_year,
            })

    print(f"Сохранено в applications.csv ({len(applications)} строк).")
    return applications


if __name__ == "__main__":
    main()
