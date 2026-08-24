"""The subsequent memory effect: recalled minus forgotten, averaged over trials.

Produces the same figure twice -- once from ocular-corrected epochs and once from
uncorrected ones. The comparison is the point. Blinks are roughly 100x the amplitude
of the effect and are not guaranteed to be independent of the recall label, so an
effect that appears only in the uncorrected panel is an artifact of eyelids.
"""

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from peers.eeg import DERIV_ROOT

mne.set_log_level("ERROR")

OUT = Path(__file__).resolve().parents[1] / "findings"
OUT.mkdir(exist_ok=True)

#: Midline equivalents on the EGI HydroCel-129 net. The SME is classically reported
#: as a frontal and parietal positivity for later-recalled items.
MIDLINE = {"Fz": "E11", "FCz": "E6", "Cz": "Cz", "Pz": "E62"}

#: The window the SME literature reports the effect in.
EFFECT_WINDOW = (0.4, 0.8)


def load_condition(tag: str) -> mne.Epochs:
    files = sorted(DERIV_ROOT.glob(f"*/*_{tag}_epo.fif"))
    if not files:
        raise FileNotFoundError(f"No cached epochs for '{tag}'. Run scripts/build_epochs.py.")
    return mne.concatenate_epochs(
        [mne.read_epochs(f, preload=True, verbose="ERROR") for f in files], verbose="ERROR"
    )


def summarise(epochs: mne.Epochs, label: str) -> dict:
    rec = epochs["recalled"].average()
    forg = epochs["forgotten"].average()
    diff = mne.combine_evoked([rec, forg], weights=[1, -1])
    print(f"{label}: {len(epochs['recalled'])} recalled, {len(epochs['forgotten'])} forgotten")
    return {"recalled": rec, "forgotten": forg, "difference": diff}


conditions = {tag: summarise(load_condition(tag), tag) for tag in ("ocular-removed", "ocular-kept")}

# --- midline waveforms, corrected vs uncorrected -------------------------------------
fig, axes = plt.subplots(len(MIDLINE), 2, figsize=(11, 9), sharex=True, sharey="row")
for col, (tag, ev) in enumerate(conditions.items()):
    for row, (label, ch) in enumerate(MIDLINE.items()):
        ax = axes[row, col]
        if ch not in ev["recalled"].ch_names:
            ax.set_visible(False)
            continue
        idx = ev["recalled"].ch_names.index(ch)
        t = ev["recalled"].times
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.axvspan(*EFFECT_WINDOW, color="grey", alpha=0.15)
        ax.plot(t, ev["recalled"].data[idx] * 1e6, color="tab:green", label="later recalled")
        ax.plot(t, ev["forgotten"].data[idx] * 1e6, color="tab:red", label="later forgotten")
        ax.plot(t, ev["difference"].data[idx] * 1e6, color="k", lw=1.6, label="difference")
        ax.set_ylabel(f"{label}\nµV")
        if row == 0:
            ax.set_title(tag)
            ax.legend(fontsize=7, loc="upper right")
        if row == len(MIDLINE) - 1:
            ax.set_xlabel("Time from word onset (s)")
        ax.set_xlim(-0.2, 1.6)
fig.suptitle("Subsequent memory effect at midline electrodes", y=0.995)
fig.tight_layout()
fig.savefig(OUT / "sme_midline.png", dpi=150)

# --- scalp topography of the difference ----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, (tag, ev) in zip(axes, conditions.items()):
    ev["difference"].plot_topomap(
        times=[np.mean(EFFECT_WINDOW)],
        average=EFFECT_WINDOW[1] - EFFECT_WINDOW[0],
        axes=ax,
        colorbar=False,
        show=False,
        time_format="",
    )
    ax.set_title(f"{tag}\n{EFFECT_WINDOW[0]}-{EFFECT_WINDOW[1]} s")
fig.suptitle("Recalled minus forgotten")
fig.tight_layout()
fig.savefig(OUT / "sme_topography.png", dpi=150)

# --- numbers ---------------------------------------------------------------------------
print(f"\nMean difference (recalled - forgotten) in {EFFECT_WINDOW[0]}-{EFFECT_WINDOW[1]} s, µV:")
for tag, ev in conditions.items():
    d = ev["difference"]
    mask = (d.times >= EFFECT_WINDOW[0]) & (d.times <= EFFECT_WINDOW[1])
    vals = {lab: d.data[d.ch_names.index(ch)][mask].mean() * 1e6
            for lab, ch in MIDLINE.items() if ch in d.ch_names}
    joined = "  ".join(f"{lab} {v:+.3f}" for lab, v in vals.items())
    print(f"  {tag:>14}: {joined}   |  peak abs across scalp "
          f"{np.abs(d.data[:, mask].mean(axis=1)).max() * 1e6:.3f}")

print(f"\nsaved {OUT}/sme_midline.png")
print(f"saved {OUT}/sme_topography.png")
