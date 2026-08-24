"""Band effects in sub-windows of the encoding epoch, from the cached contrasts.

The single band mean reported by `plot_spectral_sme.py` averages 300-1600 ms, and R4 records
that the low-frequency effect is not constant across that window. This script resolves the
window without recomputing anything, by reading the per-subject contrast arrays that
`plot_spectral_sme.py` caches.

Every number in R4's sub-window table and every claim about when the effect starts comes from
here. Reading a time course off a colour map is not a measurement.

Usage:  PYTHONPATH=src uv run python scripts/spectral_subwindows.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "findings"
N_PERMUTATIONS = 10_000
SEED = 0

#: Narrower than the four conventional bands, because R4 found the early effect confined to
#: 2-4 Hz while 4-8 Hz is flat over the same window. Splitting theta is the only way to see it.
BANDS = {
    "delta 2-4": (2, 4),
    "theta 4-8": (4, 8),
    "alpha 8-13": (8, 13),
    "beta 13-30": (13, 30),
    "gamma 30-100": (30, 100),
}

#: Equal halves of the pre-registered 300-1600 ms window (D26). The split is deliberately not
#: placed where the effect reverses: choosing a boundary from the data would make any p value
#: below it uninterpretable.
WINDOWS = {
    "full 0.30-1.60": (0.30, 1.60),
    "early 0.30-0.95": (0.30, 0.95),
    "late 0.95-1.60": (0.95, 1.60),
}

#: 200 ms steps across the whole epoch, for the time course rather than for testing.
STEP = 0.2


def sign_flip(values: np.ndarray, rng: np.random.Generator) -> float:
    """Two-sided p for a per-subject effect, treating subject as the unit of observation."""
    observed = values.mean()
    flips = rng.choice([-1.0, 1.0], size=(N_PERMUTATIONS, len(values)))
    return float((np.abs((flips * values).mean(axis=1)) >= abs(observed)).mean())


def band_window(contrasts, freqs, times, band, window) -> np.ndarray:
    """Per-subject mean contrast inside one band and one time window."""
    in_band = (freqs >= band[0]) & (freqs < band[1])
    in_window = (times >= window[0]) & (times <= window[1])
    return contrasts[:, in_band][:, :, in_window].mean(axis=(1, 2))


def main() -> None:
    rows, course = [], []

    for path in sorted(OUT.glob("spectral_contrasts_*.npz")):
        tag = path.stem.replace("spectral_contrasts_", "")
        cache = np.load(path, allow_pickle=False)
        contrasts, times, freqs = cache["contrasts"], cache["times"], cache["freqs"]
        rng = np.random.default_rng(SEED)

        print(f"\n{tag}  ({contrasts.shape[0]} subjects)")
        for wname, window in WINDOWS.items():
            print(f"  {wname}")
            for bname, band in BANDS.items():
                per_subject = band_window(contrasts, freqs, times, band, window)
                effect = float(per_subject.mean())
                p = sign_flip(per_subject, rng)
                negative = int((per_subject < 0).sum())
                print(f"    {bname:<14} {effect:+9.4f}  p={p:.4f}  "
                      f"{negative}/{len(per_subject)} subjects negative")
                rows.append({"condition": tag, "window": wname, "band": bname,
                             "n_subjects": len(per_subject), "effect": effect, "p": p,
                             "n_negative": negative})

        # Time course of the low frequencies, which is where the reversal lives.
        low = (freqs >= 2) & (freqs < 8)
        for start in np.arange(float(times[0]), float(times[-1]) - STEP, STEP):
            in_window = (times >= start) & (times < start + STEP)
            if not in_window.any():
                continue
            per_subject = contrasts[:, low][:, :, in_window].mean(axis=(1, 2))
            course.append({"condition": tag, "t_start": round(float(start), 2),
                           "t_end": round(float(start + STEP), 2),
                           "effect": float(per_subject.mean()),
                           "p": sign_flip(per_subject, rng)})

    pd.DataFrame(rows).to_csv(OUT / "spectral_subwindows.csv", index=False)
    pd.DataFrame(course).to_csv(OUT / "spectral_timecourse_2-8Hz.csv", index=False)
    print(f"\nsaved {OUT}/spectral_subwindows.csv and spectral_timecourse_2-8Hz.csv")


if __name__ == "__main__":
    main()
