"""The subsequent memory effect in oscillatory power.

The ERP difference wave is the textbook demonstration of the SME, but it comes from short,
tightly controlled paradigms. In free recall the effect is reported predominantly in
oscillatory power, and spectral measures are far more robust to the noise in 90-minute
recordings: one large voltage transient moves an average of raw voltage directly, but barely
moves log power in a band.

Normalisation follows the published treatment of this dataset family [D26]: power is
log-transformed, then z-scored within session for each electrode and frequency, using the mean
and standard deviation across all events in that session. No pre-stimulus baseline is
subtracted. This removes between-session and between-electrode scale differences -- electrode
impedance and amplifier gain vary far more between sessions than the effect being measured --
without imposing a baseline window.

Contrasts are computed within subject and then averaged across subjects, never by pooling
epochs: between-subject differences are much larger than the memory effect, so pooling would
let subject identity dominate.

Memory note: a session's time-frequency array is ~1.4 GB, so sessions are reduced to
per-condition means as they are processed rather than concatenated. Do not run this while a
dataset build is in progress.
"""

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from peers.eeg import DERIV_ROOT

mne.set_log_level("ERROR")

OUT = Path(__file__).resolve().parents[1] / "findings"

#: Log-spaced so low frequencies, where the classic bands are narrow, are sampled densely.
FREQS = np.logspace(np.log10(2), np.log10(100), 30)
N_CYCLES = FREQS / 2.0
DECIM = 4

BANDS = {
    "theta (4-8 Hz)": (4, 8),
    "alpha (8-13 Hz)": (8, 13),
    "beta (13-30 Hz)": (13, 30),
    "gamma (30-100 Hz)": (30, 100),
}

#: Encoding window. [Long 2014] and [Broitman 2025] analyse 0-1600 ms on this paradigm;
#: [Weidemann 2021] truncates to 300-1600 ms to isolate item-specific effects. We follow the
#: latter, starting after the earliest visual response.
ANALYSIS_WINDOW = (0.3, 1.6)

CONDITIONS = ("ocular-removed", "ocular-kept")
N_PERMUTATIONS = 1000


def _trial_key(metadata: pd.DataFrame) -> list[tuple]:
    """Identify a trial by its presentation, so it can be found in both ocular conditions."""
    return list(zip(metadata["session"], metadata["trial"], metadata["item_num"]))


def common_trials(subject: str) -> set | None:
    """Trials surviving AutoReject in both ocular conditions.

    AutoReject runs independently per condition and keeps different trials, so comparing the
    conditions as built would confound the effect of ocular correction with the effect of
    analysing different subsets.
    """
    key_sets = []
    for tag in CONDITIONS:
        files = sorted(DERIV_ROOT.glob(f"{subject}/*_{tag}_epo.fif"))
        if not files:
            return None
        metadata = pd.concat(
            [mne.read_epochs(f, preload=False, verbose="ERROR").metadata for f in files]
        )
        key_sets.append(set(_trial_key(metadata)))
    return set.intersection(*key_sets)


def session_power(epochs: mne.Epochs) -> np.ndarray:
    """Log power, z-scored within this session, averaged over channels.

    Returns (n_epochs, n_freqs, n_times). The z-score uses the mean and standard deviation
    over all events and time points in the session, separately for each electrode and
    frequency, matching the normalisation used in the published analyses of this dataset.
    """
    tfr = epochs.compute_tfr(
        method="morlet", freqs=FREQS, n_cycles=N_CYCLES,
        return_itc=False, average=False, decim=DECIM, n_jobs=2, verbose="ERROR",
    )
    data = np.log10(tfr.get_data() + 1e-30)          # (n_epochs, n_ch, n_freqs, n_times)
    mean = data.mean(axis=(0, 3), keepdims=True)
    std = data.std(axis=(0, 3), keepdims=True)
    return ((data - mean) / np.where(std == 0, 1.0, std)).mean(axis=1)


def subject_condition_means(subject: str, tag: str, restrict_to: set) -> tuple | None:
    """Mean z-scored power for recalled and for forgotten trials, for one subject.

    Both conditions are returned rather than only their difference, so that a later question
    about which condition drives an effect can be answered from the cache without recomputing
    the time-frequency decomposition.
    """
    totals = {True: None, False: None}
    counts = {True: 0, False: 0}
    times = None

    for path in sorted(DERIV_ROOT.glob(f"{subject}/*_{tag}_epo.fif")):
        epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
        keep = [i for i, k in enumerate(_trial_key(epochs.metadata)) if k in restrict_to]
        if len(keep) < 2:
            continue
        epochs = epochs[keep]
        labels = epochs.metadata["recalled"].to_numpy()
        if len(np.unique(labels)) < 2:
            continue

        power = session_power(epochs)
        if times is None:
            times = epochs.times[::DECIM][: power.shape[-1]]
        for outcome in (True, False):
            block = power[labels == outcome]
            if not len(block):
                continue
            summed = block.sum(axis=0)
            totals[outcome] = summed if totals[outcome] is None else totals[outcome] + summed
            counts[outcome] += len(block)
        del power, epochs

    if not all(counts.values()):
        return None
    return totals[True] / counts[True], totals[False] / counts[False], times, counts


def main() -> None:
    subjects = sorted(p.name for p in DERIV_ROOT.iterdir() if p.is_dir())
    shared = {s: common_trials(s) for s in subjects}
    incomplete = sorted(s for s, v in shared.items() if not v)
    if incomplete:
        print(f"skipping {len(incomplete)} subject(s) missing a condition: {', '.join(incomplete)}")
    subjects = [s for s in subjects if shared.get(s)]
    if not subjects:
        print("no subjects have both ocular conditions yet")
        return
    print(f"{len(subjects)} subjects; {sum(len(shared[s]) for s in subjects):,} trials "
          f"common to both ocular conditions")

    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    for tag in CONDITIONS:
        results = [(s, subject_condition_means(s, tag, shared[s])) for s in subjects]
        results = [(s, r) for s, r in results if r is not None]
        if not results:
            print(f"{tag}: no usable data")
            continue

        recalled = np.stack([r[0] for _, r in results])        # (n_subj, n_freqs, n_times)
        forgotten = np.stack([r[1] for _, r in results])
        contrasts = recalled - forgotten
        times = results[0][1][2]
        grand = contrasts.mean(axis=0)

        # The decomposition costs roughly 25 minutes over the full sample, so the per-subject
        # arrays are cached. Any later question about a different band, window or condition is
        # then answered from this file rather than by recomputing.
        np.savez_compressed(
            OUT / f"spectral_contrasts_{tag}.npz",
            contrasts=contrasts, recalled=recalled, forgotten=forgotten,
            times=times, freqs=FREQS,
            subjects=np.array([s for s, _ in results]),
            n_recalled=np.array([r[3][True] for _, r in results]),
            n_forgotten=np.array([r[3][False] for _, r in results]),
        )

        fig, ax = plt.subplots(figsize=(8, 4.5))
        limit = np.abs(grand).max()
        mesh = ax.pcolormesh(times, FREQS, grand, cmap="RdBu_r",
                             vmin=-limit, vmax=limit, shading="gouraud")
        ax.set_yscale("log")
        ax.set_yticks([2, 4, 8, 13, 30, 60, 100])
        ax.set_yticklabels(["2", "4", "8", "13", "30", "60", "100"])
        ax.axvline(0, color="k", lw=0.8)
        ax.axvspan(*ANALYSIS_WINDOW, fill=False, edgecolor="k", lw=0.8, ls="--")
        ax.set_xlim(-0.5, 2.0)
        ax.set_xlabel("Time from word onset (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Subsequent memory effect in power ({tag}, n={len(results)} subjects)")
        fig.colorbar(mesh, ax=ax, label="z-scored log power\n(recalled - forgotten)")
        fig.tight_layout()
        fig.savefig(OUT / f"spectral_sme_{tag}.png", dpi=150)

        window = (times >= ANALYSIS_WINDOW[0]) & (times <= ANALYSIS_WINDOW[1])
        print(f"\n{tag}  (n={len(results)} subjects, {ANALYSIS_WINDOW[0]}-{ANALYSIS_WINDOW[1]} s)")
        print(f"  {'band':<20} {'effect':>10} {'p':>8}")
        rows = []
        for name, (low, high) in BANDS.items():
            band = (FREQS >= low) & (FREQS < high)
            if not band.any():
                continue
            per_subject = contrasts[:, band][:, :, window].mean(axis=(1, 2))
            observed = per_subject.mean()
            # Sign-flip test: under the null a subject's effect is equally likely to point
            # either way. Treating subject as the unit of observation is what generalising
            # to new people requires.
            flips = rng.choice([-1, 1], size=(N_PERMUTATIONS, len(per_subject)))
            null = (flips * per_subject).mean(axis=1)
            p = float((np.abs(null) >= abs(observed)).mean())
            print(f"  {name:<20} {observed:+10.4f} {p:8.3f}")
            rows.append({"condition": tag, "band": name, "n_subjects": len(per_subject),
                         "effect": observed, "p": p})

        pd.DataFrame(rows).to_csv(OUT / f"spectral_sme_{tag}.csv", index=False)
        print(f"  saved {OUT}/spectral_sme_{tag}.png and .csv")


if __name__ == "__main__":
    main()
