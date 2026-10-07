"""Plot benchmarks/step_time.csv to benchmarks/step_time.png: time per training step of each iteration."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BLUE, ORANGE, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#1a1a19", "#6b6b66", "#fcfcfb"
here = Path(__file__).parent
rows = [r for r in csv.DictReader(open(here / "step_time.csv")) if r["ms_per_step"]]
panels = {"bench": "Bench config (same model & data as the HF baseline)",
          "machine-translation": "Machine-translation config (image + bytes encoders, batch 64)"}

fig, axes = plt.subplots(len(panels), 1, figsize=(10, 7.5), facecolor=SURFACE,
                         gridspec_kw={"height_ratios": [sum(r["config"] == c for r in rows) for c in panels]})
for ax, (config, title) in zip(axes, panels.items(), strict=True):
    panel = [r for r in rows if r["config"] == config]
    baseline = float(panel[0]["ms_per_step"]) if config != "bench" else float(rows[0]["ms_per_step"])
    labels = [f"{r['iteration']}. {r['short']}" for r in panel][::-1]
    times = [float(r["ms_per_step"]) for r in panel][::-1]
    kept = [r["kept"] == "yes" for r in panel][::-1]
    hf = [r["stack"].startswith("HF") for r in panel][::-1]
    for y, (time, keep, is_hf) in enumerate(zip(times, kept, hf, strict=True)):
        color = ORANGE if is_hf else BLUE
        ax.barh(y, time, height=0.6, color=color if keep else "none", edgecolor=color, linewidth=1.5,
                hatch=None if keep else "///")
        ax.text(time + baseline * 0.01, y, f"{time:.0f} ms  ({baseline / time:.1f}x)", va="center", fontsize=9,
                color=INK)
    ax.set_yticks(range(len(labels)), labels, fontsize=9, color=INK)
    ax.set_xlim(0, max(times) * 1.3)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_facecolor(SURFACE)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.grid(axis="x", color="#e4e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
axes[-1].set_xlabel("ms per training step on GB10 (lower is better); speedup vs. the panel's first row",
                    color=MUTED, fontsize=9)
fig.legend(handles=[Patch(facecolor=ORANGE, label="HF Trainer (main)"),
                    Patch(facecolor=BLUE, label="Megatron-Bridge, kept"),
                    Patch(facecolor="none", edgecolor=BLUE, hatch="///", label="tried, not kept / option")],
           loc="upper left", bbox_to_anchor=(0.005, 0.955), frameon=False, fontsize=9, ncol=3)
fig.suptitle("WeLT time per training step, by optimization iteration", x=0.01, ha="left", fontsize=12, color=INK)
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig(here / "step_time.png", dpi=150, facecolor=SURFACE)
print(f"Saved {here / 'step_time.png'}")
