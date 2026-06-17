

from __future__ import annotations

import datetime
import json
import uuid
from json.decoder import JSONDecodeError
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from tools import (
    calculate,
    compare_periods,
    get_fx_rate,
    get_inflation,
    get_key_rate,
    get_unemployment,
)

TOOLS_IMPL = {
    "get_fx_rate": get_fx_rate,
    "get_key_rate": get_key_rate,
    "get_inflation": get_inflation,
    "get_unemployment": get_unemployment,
    "calculate": calculate,
    "compare_periods": compare_periods,
}

TRACE_PATH = Path(__file__).resolve().parent / "trace.jsonl"
CACHE_STATS = {"hits": 0, "misses": 0}


def _log_event(run_id: str, entry: dict) -> None:
    """Дословно как в agent.py — одна строка-событие в trace.jsonl."""
    record = {
        "run_id": run_id,
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        **entry,
    }
    with open(TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _exec_one(tc, cache: Optional[dict] = None):
    """Дословная копия agent._exec_one — тот же путь обработки ошибок."""
    name = tc.function.name
    try:
        args = json.loads(tc.function.arguments or "{}")
    except JSONDecodeError as e:
        return tc, {}, {"error": f"битый json аргументов: {e}"}

    fn = TOOLS_IMPL.get(name)
    if fn is None:
        return tc, args, {"error": f"неизвестный инструмент: {name}"}

    try:
        obs = fn(**args)
    except TypeError as e:
        return tc, args, {
            "error": f"плохие аргументы для {name}: {e}. Expected: {fn.__annotations__}"
        }
    except Exception as e:
        return tc, args, {"error": f"{type(e).__name__}: {e}"}
    return tc, args, obs


def _fake_call(name: str, arguments: str, call_id: str = "call_x"):
    """Сэмулировать tool_call, как его прислала бы модель."""
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )

FAULTS = [
    (
        "битый JSON в аргументах",
        _fake_call("get_fx_rate", '{"currency": "USD", "on_date": "2022-01-15"'),  # нет }
    ),
    (
        "галлюцинация инструмента",
        _fake_call("get_gdp", '{"year": 2024, "quarter": 3}'),  # такого инструмента нет
    ),
    (
        "инструмент упал на плохих аргументах",
        _fake_call("get_fx_rate", '{"ccy": "USD"}'),  # неверное имя параметра -> TypeError
    ),
    (
        "ошибка внутри инструмента (нет данных)",
        _fake_call("compare_periods", '{"metric": "cpi", "period_a": "2026-04", "period_b": "2026-03"}'),
    ),
]


def main() -> None:
    run_id = "diag_" + uuid.uuid4().hex[:8]
    print(f"run_id = {run_id}\nпишу сбойные шаги в {TRACE_PATH.name}\n")
    for step, (label, tc) in enumerate(FAULTS, start=1):
        _, args, obs = _exec_one(tc)
        entry = {"step": step, "call": tc.function.name, "args": args, "obs": obs}
        _log_event(run_id, entry)
        print(f"[{step}] {label}")
        print(f"    call: {tc.function.name}  raw_args: {tc.function.arguments}")
        print(f"    obs : {json.dumps(obs, ensure_ascii=False)}\n")
    _log_event(run_id, {"step": len(FAULTS) + 1, "final": None,
                         "error": "диагностический прогон: см. obs выше"})
    print("Готово. Строки добавлены в trace.jsonl (режим append).")


if __name__ == "__main__":
    main()
