

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from fake_broken import FAKE_BROKEN

N = int(os.environ.get("CRITIC_N", "10"))
TEMPS = (0.0, 0.7)
OUT = Path(__file__).resolve().parent / "critic.json"


def measure() -> dict:
    rows = []
    for case in FAKE_BROKEN:
        row = {"label": case["label"]}
        for t in TEMPS:
            false_accepts = 0
            for _ in range(N):
                try:
                    v = critic(case["question"], case["plan"], case["answers"], temperature=t)
                    if v.ok:  # ok=True на заведомо битом кейсе = ложное принятие
                        false_accepts += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"    [warn] {case['label']} T={t}: {exc}")
            row[f"t{t}"] = false_accepts
            print(f"  [{case['label']}] T={t}: ложных принятий {false_accepts}/{N}")
        rows.append(row)

    result = {"N": N, "temps": list(TEMPS), "rows": rows}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n| Битый кейс | T=0.0 | T=0.7 |")
    print("|---|---|---|")
    for r in rows:
        print(f"| {r['label']} | {r['t0.0']}/{N} | {r['t0.7']}/{N} |")
    print(f"\nСохранено: {OUT}")
    return result


if __name__ == "__main__":
    print(f"Замер критики, N={N} на каждую температуру (~{len(FAKE_BROKEN)*2*N} запросов)...")
    measure()
