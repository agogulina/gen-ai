
import csv
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["font.family"] = "DejaVu Sans"

# данные
rows = []
with open("applications.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)

cities = [r["city"] for r in rows]
specialities = [r["speciality"] for r in rows]

n = len(rows)

# Города
city_counts = Counter(cities)
labels_c, vals_c = zip(*sorted(city_counts.items(), key=lambda x: -x[1]))
pcts_c = [v / n * 100 for v in vals_c]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(labels_c, vals_c, color="#4C72B0", edgecolor="white", linewidth=0.6)
ax.set_title("Распределение заявок по городам", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Количество заявок")
ax.set_xlabel("Город")
plt.xticks(rotation=30, ha="right", fontsize=9)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

for bar, pct in zip(bars, pcts_c):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        f"{pct:.0f}%",
        ha="center", va="bottom", fontsize=8, color="#333333"
    )

ax.axhline(n * 0.40, color="red", linestyle="--", linewidth=1, alpha=0.7, label="40% порог")
ax.legend(fontsize=9)
ax.set_ylim(0, max(vals_c) * 1.25)

plt.tight_layout()
plt.savefig("cities.png", dpi=150)
plt.close()
print("Сохранено: cities.png")

# Специальности
spec_counts = Counter(specialities)
labels_s, vals_s = zip(*sorted(spec_counts.items(), key=lambda x: -x[1]))
pcts_s = [v / n * 100 for v in vals_s]

fig, ax = plt.subplots(figsize=(13, 6))
bars = ax.bar(labels_s, vals_s, color="#DD8452", edgecolor="white", linewidth=0.6)
ax.set_title("Распределение заявок по специальностям", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Количество заявок")
ax.set_xlabel("Специальность")
plt.xticks(rotation=30, ha="right", fontsize=9)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

for bar, pct in zip(bars, pcts_s):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{pct:.0f}%",
        ha="center", va="bottom", fontsize=8, color="#333333"
    )

ax.axhline(n * 0.35, color="red", linestyle="--", linewidth=1, alpha=0.7, label="35% порог")
ax.legend(fontsize=9)
ax.set_ylim(0, max(vals_s) * 1.25)

plt.tight_layout()
plt.savefig("specialities.png", dpi=150)
plt.close()
print("Сохранено: specialities.png")

max_city_pct = max(pcts_c)
max_spec_pct = max(pcts_s)
print(f"\nМаксимальная доля одного города: {max_city_pct:.1f}% (порог 40%)")
print(f"Максимальная доля одной специальности: {max_spec_pct:.1f}% (порог 35%)")
print(f"Всего заявок: {n}/50")

