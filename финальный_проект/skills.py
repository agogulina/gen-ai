
from __future__ import annotations

import re

# группы эквивалентных терминов
_SYNONYM_GROUPS: list[set[str]] = [
    {"kubernetes", "k8s"},
    {"javascript", "js", "ecmascript"},
    {"typescript", "ts"},
    {"python", "py"},
    {"machine learning", "ml"},
    {"deep learning", "dl"},
    {"artificial intelligence", "ai"},
    {"natural language processing", "nlp"},
    {"rest", "restful", "rest api", "restful api", "api"},
    {"ci/cd", "cicd", "ci cd", "continuous integration", "continuous delivery",
     "jenkins", "gitlab ci", "github actions", "circleci"},
    {"sql", "mysql", "postgresql", "postgres", "tsql", "t-sql", "pl/sql", "plsql",
     "ms sql", "sql server", "oracle sql"},
    {"nosql", "mongodb", "mongo", "cassandra", "dynamodb"},
    {"aws", "amazon web services", "ec2", "s3", "lambda"},
    {"gcp", "google cloud", "google cloud platform"},
    {"azure", "microsoft azure"},
    {"docker", "containers", "containerization"},
    {"react", "react.js", "reactjs"},
    {"node", "node.js", "nodejs"},
    {"angular", "angular.js", "angularjs"},
    {"vue", "vue.js", "vuejs"},
    {"c#", "csharp", "c sharp", ".net", "dotnet", "asp.net"},
    {"c++", "cpp", "cplusplus"},
    {"golang", "go lang", "go"},
    {"microservices", "microservice", "micro services", "micro-service architecture"},
    {"message queue", "kafka", "rabbitmq", "event-driven", "event driven",
     "pub/sub", "pubsub"},
    {"agile", "scrum", "kanban", "sprint"},
    {"tdd", "test driven development", "test-driven"},
    {"qa", "quality assurance", "testing", "test automation", "automation testing"},
    {"selenium", "webdriver", "selenium webdriver"},
    {"data analysis", "data analytics", "analytics", "data analyst"},
    {"power bi", "powerbi", "tableau", "looker", "bi"},
    {"excel", "microsoft excel", "spreadsheets", "ms excel"},
    {"etl", "ssis", "data pipeline", "data pipelines"},
    {"linux", "unix", "bash", "shell scripting", "shell"},
    {"git", "github", "gitlab", "bitbucket", "version control"},
    {"project management", "project manager", "pmp", "stakeholder management"},
    {"requirements gathering", "requirements analysis", "business analysis",
     "business analyst", "systems analysis"},
    {"communication", "stakeholder communication", "interpersonal"},
]

_INDEX: dict[str, set[str]] = {}
for grp in _SYNONYM_GROUPS:
    for t in grp:
        _INDEX.setdefault(t, set()).update(grp)


def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./ ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def expand_skill(skill: str) -> set[str]:
    base = _norm(skill)
    variants = {base}
    if base in _INDEX:
        variants |= {_norm(x) for x in _INDEX[base]}
    if base.endswith("s") and len(base) > 3:
        variants.add(base[:-1])
    else:
        variants.add(base + "s")
    return {v for v in variants if v}


def skill_present(skill: str, resume_norm: str) -> bool:
    for v in expand_skill(skill):
        if not v:
            continue
        if re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", resume_norm):
            return True
    return False


def normalize_resume(resume_text: str) -> str:
    return _norm(resume_text)


GENERIC_SKILLS = {
    "communication", "teamwork", "team work", "leadership", "problem solving",
    "time management", "collaboration", "interpersonal", "organization",
    "stakeholder communication", "requirements gathering", "requirements analysis",
    "systems analysis", "business analysis", "documentation", "technical documentation",
    "software development", "application development", "development", "testing",
    "architecture", "design", "analysis", "reporting", "operational stability",
    "debugging", "maintenance", "support", "research", "planning", "management",
    "cloud technologies", "cloud", "software design patterns",
}


def is_generic(skill: str) -> bool:
    return _norm(skill) in GENERIC_SKILLS


if __name__ == "__main__":
    r = normalize_resume("Built microservices in Golang, deployed on k8s. Wrote REST APIs. Used Postgres.")
    for sk in ["Kubernetes", "Go", "RESTful API", "SQL", "Kafka"]:
        print(f"{sk:14} -> {skill_present(sk, r)}")
