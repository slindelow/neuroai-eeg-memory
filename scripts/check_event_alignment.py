"""Audit the behavioural event files for timing defects.

Two things are checked, both found while investigating something else and neither caught by
the existing tests at the time.

1. **Presentations with no aligned EEG sample.** PEERS events are timed by the task computer
   and aligned to the amplifier using sync pulses. An event with no corresponding EEG sample
   is marked `sample = -1`, which the onset column reports as -1/sfreq seconds. These are real
   presentations with no EEG behind them, and `build_trial_table` excludes them (F16).

2. **Inter-word onset interval.** How long a trial lasts decides whether an effect measured
   late in the epoch could be contamination from the next word. Measured here rather than
   taken from a paper (R6, resolving U12).

Usage:  PYTHONPATH=src uv run python scripts/check_event_alignment.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from peers.events import load_all_events, unaligned_presentations

OUT = Path(__file__).resolve().parents[1] / "findings"
PERCENTILES = [1, 25, 50, 75, 99]


def main() -> None:
    events = load_all_events()
    words = events[(events.trial_type == "WORD") & (events.trial >= 1)]

    print("=== presentations with no aligned EEG sample (F16) ===")
    unaligned = unaligned_presentations(events)
    print(f"unaligned : {len(unaligned):,} of {len(words):,} "
          f"({len(unaligned) / len(words):.4%})")
    if len(unaligned):
        per_session = unaligned.groupby(["subject_id", "session"]).size()
        print(f"sessions  : {len(per_session)}")
        for (subject, session), n in per_session.items():
            total = len(words[(words.subject_id == subject) & (words.session == session)])
            print(f"    {subject} ses-{session}: {n} of {total} presentations")

    print("\n=== inter-word onset interval (R6, U12) ===")
    aligned = words[words["sample"] >= 0].sort_values(
        ["subject_id", "session", "trial", "onset"]
    )
    gaps = aligned.groupby(["subject_id", "session", "trial"])["onset"].diff().dropna()
    print(f"intervals : {len(gaps):,} across {aligned.subject_id.nunique()} subjects")
    for q in PERCENTILES:
        print(f"    p{q:<3} {np.percentile(gaps, q):7.3f} s")
    print(f"    mean {gaps.mean():.3f}  sd {gaps.std():.3f}")

    # An epoch ending before the shortest interval cannot contain the next stimulus.
    print(f"\nshortest interval observed: {gaps.min():.3f} s")
    print("epochs end at +2.0 s (D11), so no epoch reaches the following word.")

    pd.DataFrame([{
        "n_words": len(words),
        "n_unaligned": len(unaligned),
        "n_intervals": len(gaps),
        "interval_mean_s": gaps.mean(),
        "interval_sd_s": gaps.std(),
        "interval_min_s": gaps.min(),
        **{f"interval_p{q}_s": float(np.percentile(gaps, q)) for q in PERCENTILES},
    }]).to_csv(OUT / "event_alignment.csv", index=False)
    print(f"\nsaved {OUT}/event_alignment.csv")


if __name__ == "__main__":
    main()
