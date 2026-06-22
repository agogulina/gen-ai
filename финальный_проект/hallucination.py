"""
Проверка галлюцинаций (обязательный блок рубрики).

Матчер обязан подтверждать совпавшие навыки ДОСЛОВНЫМИ цитатами из резюме.
Здесь мы проверяем, что каждая цитата действительно присутствует в тексте
резюме (ghost-quote), и что заявленный навык реально упоминается (ghost-skill).
Возвращаем и «очищенную» оценку, и числовые метрики для отчёта.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from schema import Evidence, FitAssessment

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    # нормализация для устойчивого сравнения: нижний регистр, схлопнутые пробелы
    return _WS.sub(" ", (text or "").lower()).strip()


def quote_is_grounded(quote: str, resume_text: str) -> bool:
    """Цитата считается подтверждённой, если дословно (после нормализации
    пробелов/регистра) встречается в резюме. Допускаем небольшую обрезку
    хвостовых знаков препинания."""
    nq = _norm(quote).strip(" .,:;—-")
    nr = _norm(resume_text)
    if not nq:
        return False
    if nq in nr:
        return True
    # запасной критерий: совпадает длинная подстрока (>=85% слов подряд),
    # чтобы не штрафовать за один лишний пробел/символ внутри
    words = nq.split()
    if len(words) >= 6:
        head = " ".join(words[: max(6, int(len(words) * 0.85))])
        return head in nr
    return False


@dataclass
class GhostReport:
    total_evidence: int = 0
    ghost_quotes: int = 0
    ghost_skills: int = 0
    grounded_evidence: int = 0
    ghost_examples: list[str] = field(default_factory=list)
    had_any_ghost: bool = False

    def as_dict(self) -> dict:
        return {
            "total_evidence": self.total_evidence,
            "grounded_evidence": self.grounded_evidence,
            "ghost_quotes": self.ghost_quotes,
            "ghost_skills": self.ghost_skills,
            "had_any_ghost": self.had_any_ghost,
            "ghost_examples": self.ghost_examples[:3],
        }


def verify_assessment(
    assessment: FitAssessment, resume_text: str
) -> tuple[FitAssessment, GhostReport]:
    """Проверить доказательства и при необходимости ПОНИЗИТЬ оценку.

    Логика: цитаты-призраки выкидываем. Навык считается подтверждённым только
    если на него есть хотя бы одна grounded-цитата. Если после чистки у вердикта
    «Good Fit» не осталось ни одного подтверждённого навыка — понижаем до
    «Potential Fit» (а если и навыков-упоминаний нет — до «No Fit»).
    """
    rep = GhostReport(total_evidence=len(assessment.evidence))
    nr = _norm(resume_text)

    clean_ev: list[Evidence] = []
    grounded_skills: set[str] = set()
    for ev in assessment.evidence:
        if quote_is_grounded(ev.quote, resume_text):
            rep.grounded_evidence += 1
            clean_ev.append(ev)
            grounded_skills.add(ev.skill.lower())
        else:
            rep.ghost_quotes += 1
            rep.ghost_examples.append(ev.quote[:80])

    # ghost-skill: заявленный matched_skill, которого нет в тексте резюме вовсе
    clean_matched: list[str] = []
    for sk in assessment.matched_skills:
        if _norm(sk) and _norm(sk) in nr:
            clean_matched.append(sk)
        else:
            rep.ghost_skills += 1
            rep.ghost_examples.append(f"skill:{sk}")

    rep.had_any_ghost = (rep.ghost_quotes > 0) or (rep.ghost_skills > 0)

    # навыки, реально подтверждённые grounded-цитатой, всегда оставляем
    for ev in clean_ev:
        if ev.skill.lower() not in {m.lower() for m in clean_matched}:
            clean_matched.append(ev.skill)

    # пересборка оценки с понижением при недостатке доказательств
    new_fit = assessment.fit
    if assessment.fit == "Good Fit" and not grounded_skills:
        new_fit = "Potential Fit" if clean_matched else "No Fit"
    elif assessment.fit == "Potential Fit" and not clean_matched and not grounded_skills:
        new_fit = "No Fit"

    new_conf = assessment.confidence
    if new_fit != assessment.fit:
        new_conf = round(min(assessment.confidence, 0.5), 3)

    if new_fit == "No Fit":
        matched_out, evidence_out = [], []
    else:
        matched_out, evidence_out = clean_matched, clean_ev

    fixed = FitAssessment(
        fit=new_fit,
        confidence=new_conf,
        matched_skills=matched_out,
        missing_skills=assessment.missing_skills,
        evidence=evidence_out,
        rationale=assessment.rationale,
    )
    return fixed, rep


def aggregate_ghost(reports: list[GhostReport]) -> dict:
    """Сводные метрики галлюцинаций по всему прогону (для output/ и отчёта)."""
    n = len(reports)
    with_ghost = sum(1 for r in reports if r.had_any_ghost)
    tot_ev = sum(r.total_evidence for r in reports)
    tot_ghost_q = sum(r.ghost_quotes for r in reports)
    tot_ghost_s = sum(r.ghost_skills for r in reports)
    return {
        "assessments": n,
        "assessments_with_any_ghost": with_ghost,
        "share_with_ghost": round(with_ghost / n, 3) if n else 0.0,
        "total_evidence": tot_ev,
        "ghost_quotes": tot_ghost_q,
        "ghost_skills": tot_ghost_s,
        "ghost_quote_rate": round(tot_ghost_q / tot_ev, 3) if tot_ev else 0.0,
    }


if __name__ == "__main__":
    resume = "Developed REST API tests in Python. Used Jenkins for CI."
    a = FitAssessment(
        fit="Good Fit", confidence=0.9, matched_skills=["Python", "Kubernetes"],
        evidence=[
            Evidence(skill="Python", quote="Developed REST API tests in Python"),
            Evidence(skill="Kubernetes", quote="Led a team of 10 Kubernetes engineers"),  # ghost
        ],
        rationale="x",
    )
    fixed, rep = verify_assessment(a, resume)
    print("ghost report:", rep.as_dict())
    print("fit after verify:", fixed.fit, "| matched:", fixed.matched_skills)
