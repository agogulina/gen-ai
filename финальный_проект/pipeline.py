

from __future__ import annotations

from agent import FIT_THRESHOLD, run_matcher_agent
from critic import skeptic_review
from hallucination import GhostReport, verify_assessment
from schema import FIT_LABELS, FIT_ORDER, FitAssessment


def assess_pair(resume_text: str, job_description: str, *, use_skeptic: bool = True) -> dict:
    out = run_matcher_agent(resume_text, job_description)
    req = out["requirements"]
    raw: FitAssessment = out["assessment"]
    coverage = out["coverage"]
    cov_label = out["coverage_label"]
    cov_ratio = coverage["ratio"]

    verified, ghost = verify_assessment(raw, resume_text)

    matched = list(dict.fromkeys([*verified.matched_skills, *coverage["present"]]))
    missing = coverage["absent"]

    base_order = max(FIT_ORDER[verified.fit], FIT_ORDER[cov_label])
    final_fit = FIT_LABELS[base_order]

    # Good Fit без доказательств недопустим
    if final_fit == "Good Fit" and not verified.evidence and not matched:
        final_fit = "Potential Fit"

    skeptic = None
    if use_skeptic:
        sv = skeptic_review(req, verified)
        skeptic = sv
        if not sv.agree and FIT_ORDER[sv.adjusted_fit] < FIT_ORDER[final_fit]:
            stepped = max(FIT_ORDER[final_fit] - 1, FIT_ORDER[sv.adjusted_fit], 0)
            final_fit = FIT_LABELS[stepped]
            if cov_ratio >= FIT_THRESHOLD and final_fit == "No Fit":
                final_fit = "Potential Fit" # при ненулевом покрытии не уводим в No Fit

    from retrieval import build_index
    idx = build_index(resume_text)
    have = {e.skill.lower() for e in verified.evidence}
    evidence = [e.model_dump() for e in verified.evidence]
    for sk in matched:
        if sk.lower() in have:
            continue
        hits = idx.search(sk, top_k=1)
        if hits and hits[0].score > 0:
            evidence.append({"skill": sk, "quote": hits[0].text[:300]})
            have.add(sk.lower())

    return {
        "requirements": req.model_dump(),
        "raw_fit": raw.fit,
        "verified_fit": verified.fit,
        "coverage": coverage,
        "coverage_label": cov_label,
        "final_fit": final_fit,
        "confidence": verified.confidence,
        "matched_skills": matched,
        "missing_skills": missing,
        "evidence": evidence,
        "rationale": verified.rationale,
        "ghost": ghost.as_dict(),
        "_ghost_obj": ghost,
        "skeptic": skeptic.model_dump() if skeptic else None,
        "n_tool_calls": out["n_tool_calls"],
        "n_steps": out["n_steps"],
    }


if __name__ == "__main__":
    rt = "Developed REST API tests in Python. Used Jenkins for CI. Selenium WebDriver."
    jd = "QA Automation Engineer: Python, Selenium, Jenkins CI/CD, 2+ years."
    import json
    r = assess_pair(rt, jd)
    r.pop("_ghost_obj", None)
    print(json.dumps(r, ensure_ascii=False, indent=2))
