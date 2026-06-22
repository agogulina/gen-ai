"""
Инструменты матчер-агента. Работают над ОДНИМ резюме: агент не получает весь
текст в промпт, а достаёт релевантные фрагменты через RAG (search_resume) и
проверяет наличие конкретного навыка (check_skill_present).

Каждый инструмент возвращает dict (как в семинарах). Реализации привязываются к
контексту конкретной пары через make_tools().
"""

from __future__ import annotations

import re

from retrieval import BM25, build_index

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_resume",
            "description": (
                "Найти в резюме кандидата фрагменты, релевантные запросу "
                "(навык, технология, обязанность). Возвращает топ-фрагменты с оценкой."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Навык/фраза для поиска, напр. 'python rest api'"},
                    "top_k": {"type": "integer", "description": "Сколько фрагментов вернуть (1..5)", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_skill_present",
            "description": (
                "Проверить, упоминается ли навык/ключевое слово в резюме дословно. "
                "Возвращает present=true/false и короткий контекст вокруг совпадения."
            ),
            "parameters": {
                "type": "object",
                "properties": {"skill": {"type": "string"}},
                "required": ["skill"],
            },
        },
    },
]


def make_tools(resume_text: str, index: BM25 | None = None):
    """Вернуть (TOOLS_IMPL, used_log) для конкретного резюме."""
    idx = index or build_index(resume_text)
    low = resume_text.lower()
    used: list[dict] = []

    def search_resume(query: str, top_k: int = 3) -> dict:
        top_k = max(1, min(int(top_k or 3), 5))
        hits = idx.search(query, top_k=top_k)
        used.append({"call": "search_resume", "query": query})
        return {
            "query": query,
            "hits": [{"score": h.score, "text": h.text} for h in hits] or [{"score": 0, "text": "(ничего не найдено)"}],
        }

    def check_skill_present(skill: str) -> dict:
        used.append({"call": "check_skill_present", "skill": skill})
        s = (skill or "").lower().strip()
        pos = low.find(s)
        if pos == -1:
            return {"skill": skill, "present": False}
        ctx = resume_text[max(0, pos - 40): pos + len(s) + 40]
        return {"skill": skill, "present": True, "context": re.sub(r"\s+", " ", ctx).strip()}

    impl = {"search_resume": search_resume, "check_skill_present": check_skill_present}
    return impl, used
