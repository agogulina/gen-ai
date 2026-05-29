
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

CURRENT_YEAR = date.today().year

CITIES = [
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Нижний Новгород",
    "Самара",
    "Краснодар",
    "Ростов-на-Дону",
    "Уфа",
    "Пермь",
    "Воронеж",
]


class Address(BaseModel):
    city: Literal[
        "Москва",
        "Санкт-Петербург",
        "Новосибирск",
        "Екатеринбург",
        "Казань",
        "Нижний Новгород",
        "Самара",
        "Краснодар",
        "Ростов-на-Дону",
        "Уфа",
        "Пермь",
        "Воронеж",
    ]
    district: str = Field(min_length=2, max_length=50)


class Application(BaseModel):
    full_name: str = Field(min_length=5, max_length=80)
    age: int = Field(ge=22, le=65)
    address: Address
    speciality: Literal[
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
    desired_course: Literal[
        "Цифровые технологии в образовании",
        "Управление проектами в здравоохранении",
        "Современные методы педагогики",
        "Психологическое консультирование",
        "Бухгалтерский учёт и налогообложение",
        "Правовое регулирование в социальной сфере",
        "Управление персоналом",
        "Информационная безопасность",
    ]
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)

    @field_validator("graduation_year")
    @classmethod
    def graduation_consistent_with_age(cls, v: int, info) -> int:
        
        age = info.data.get("age")
        if age is not None:
            birth_year = CURRENT_YEAR - age
            earliest_graduation = birth_year + 18
            if v < earliest_graduation:
                raise ValueError(
                    f"Год окончания вуза {v} противоречит возрасту {age}: "
                    f"при рождении в {birth_year} году раньше {earliest_graduation} нельзя окончить вуз."
                )
        if v > CURRENT_YEAR:
            raise ValueError(f"Год окончания вуза {v} не может быть в будущем.")
        return v
