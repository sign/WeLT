"""Plot benchmarks/step_time.csv to benchmarks/step_time.png"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

here = Path(__file__).parent
rows = list(csv.DictReader(open(here / "step_time.csv")))
labels = [f"{r['iteration']}. {r['short']}" for r in rows]
times = [float(r["ms_per_step"]) for r in rows]
colors = ["#9aa0a6" if r["stack"].startswith("HF") else "#1a73e8" if r["kept"] == "yes" else "#f9ab00"
          for r in rows]

fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 1.5))
bars = ax.barh(labels[::-1], times[::-1], color=colors[::-1])
for bar, time in zip(bars, times[::-1], strict=True):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
            f"{time:.0f} ms ({times[0] / time:.1f}x)", va="center", fontsize=9)
ax.set_xlabel("ms per training step (GB10, batch 32 x 128 words; lower is better)")
ax.set_title("WeLT training step time: HF Trainer -> Megatron-Bridge")
ax.set_xlim(0, max(times) * 1.25)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(here / "step_time.png", dpi=150)
print(f"Saved {here / 'step_time.png'}")
