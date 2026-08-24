"""Serial position curve and per-subject recall rates for the PEERS trial table.

Saves two figures to findings/. Both are validation checks as much as results: if the
serial position curve does not show primacy and recency, the labels are wrong.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from peers.events import build_trial_table, load_all_events

OUT = Path(__file__).resolve().parents[1] / "findings"
OUT.mkdir(exist_ok=True)

table = build_trial_table(load_all_events())

# --- serial position curve, pooled and per subject ---
fig, ax = plt.subplots(figsize=(7, 4.5))
for _, sub in table.groupby("subject_id"):
    curve = sub.groupby("serial_position")["recalled"].mean()
    ax.plot(curve.index, curve.values, color="grey", alpha=0.25, linewidth=1)
pooled = table.groupby("serial_position")["recalled"].mean()
ax.plot(pooled.index, pooled.values, color="tab:red", linewidth=2.5, marker="o",
        label=f"pooled (n={table['subject_id'].nunique()} subjects)")
ax.set_xlabel("Serial position within list")
ax.set_ylabel("P(recalled)")
ax.set_title("Serial position curve — primacy and recency")
ax.set_xticks(range(1, 17))
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "serial_position_curve.png", dpi=150)

# --- per-subject recall rate, showing how wide individual differences are ---
fig, ax = plt.subplots(figsize=(7, 4))
rates = table.groupby("subject_id")["recalled"].mean().sort_values()
ax.barh(range(len(rates)), rates.values, color="tab:blue", alpha=0.8)
ax.set_yticks(range(len(rates)))
ax.set_yticklabels(rates.index, fontsize=8)
ax.axvline(table["recalled"].mean(), color="tab:red", linestyle="--",
           label=f"pooled mean = {table['recalled'].mean():.2f}")
ax.set_xlabel("P(recalled)")
ax.set_xlim(0, 1)
ax.set_title("Recall rate varies enormously between participants")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "recall_rate_by_subject.png", dpi=150)

print(f"studied words : {len(table):,}")
print(f"subjects      : {table['subject_id'].nunique()}")
print(f"recall rate   : {table['recalled'].mean():.3f}")
print(f"  per-subject : {rates.min():.3f} - {rates.max():.3f}")
print(f"saved {OUT}/serial_position_curve.png")
print(f"saved {OUT}/recall_rate_by_subject.png")
