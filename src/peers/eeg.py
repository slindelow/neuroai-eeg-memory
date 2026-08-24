"""EEG preprocessing and epoching for PEERS.

Recordings are EGI HydroCel 129-channel nets at 500 Hz, referenced to Cz.
Four electrodes (E8, E25, E126, E127) sit near the eyes and are recorded as EOG.

The pipeline is deliberately runnable in two modes -- with and without ocular
correction -- because blinks are ~100x the size of the effect being measured and may
not be independent of the recall label. Any result that survives only in the
uncorrected condition is an artifact, not a finding.
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject

from peers.events import DATA_ROOT, PROJECT_DATA_ROOT

DERIV_ROOT = PROJECT_DATA_ROOT / "derivatives"

#: Electrodes placed around the eyes on the HydroCel net.
EOG_CHANNELS = ["E8", "E25", "E126", "E127"]

#: E129 is the Cz reference electrode; the standard montage calls it "Cz".
REFERENCE_CHANNEL_ALIAS = {"E129": "Cz"}

#: The recording reference. Its trace is identically zero -- it is subtracted from
#: itself -- so it must be excluded from variance-based bad-channel detection, which
#: would otherwise flag it as the most extreme outlier on the head. It is deliberately
#: kept in the data: average referencing reconstructs its true signal.
REFERENCE_CHANNEL = "Cz"

HIGHPASS_HZ = 0.5

#: Low-pass at 100 Hz rather than the 40 Hz an ERP analysis would need. The spectral
#: subsequent-memory literature reports effects in gamma as well as alpha and theta, and
#: filtering at 40 Hz would delete that band permanently -- recoverable only by
#: re-downloading and re-processing every session. Keeping 100 Hz costs nothing now and
#: leaves the band-limit as an analysis-time choice.
#:
#: Caveat to carry forward: scalp gamma is badly contaminated by muscle activity, so a
#: gamma effect here needs more scepticism than an alpha or theta one.
LOWPASS_HZ = 100.0

#: 250 Hz keeps Nyquist at 125 Hz, comfortably above the 100 Hz low-pass.
TARGET_SFREQ = 250.0

EPOCH_TMIN = -1.0
EPOCH_TMAX = 2.0
BASELINE = (-0.2, 0.0)

#: ICA is fitted on a 1 Hz high-passed copy: slow drift otherwise dominates the
#: decomposition and blink components come out mixed with it.
ICA_FIT_HIGHPASS_HZ = 1.0
ICA_N_COMPONENTS = 25

#: AutoReject search grid (Jas et al., NeuroImage 2017). Rather than one global
#: amplitude threshold, it cross-validates a per-channel threshold, then for each epoch
#: interpolates a small number of bad channels or -- if too many are bad -- drops it.
#: On this data a single global threshold costs ~40% of trials; AutoReject costs ~9%.
AUTOREJECT_N_INTERPOLATE = [1, 4, 8]
AUTOREJECT_CONSENSUS = np.linspace(0, 1, 6)


def raw_path(subject: str, session: int, root: Path = DATA_ROOT) -> Path:
    return root / subject / f"ses-{session}" / f"{subject}_ses-{session}_task-ltpFR_eeg.edf"


def load_raw(subject: str, session: int, root: Path = DATA_ROOT) -> mne.io.BaseRaw:
    """Read one session's EDF with channel types and electrode positions attached."""
    raw = mne.io.read_raw_edf(raw_path(subject, session, root), preload=True, verbose="ERROR")
    raw.rename_channels(REFERENCE_CHANNEL_ALIAS)
    raw.set_channel_types({ch: "eog" for ch in EOG_CHANNELS if ch in raw.ch_names})
    raw.set_montage("GSN-HydroCel-129", on_missing="warn", verbose="ERROR")
    return raw


def find_bad_channels(raw: mne.io.BaseRaw, z_threshold: float = 4.0) -> list[str]:
    """Flag electrodes whose variance is a gross outlier.

    Uses a median-absolute-deviation z-score on log variance, which is insensitive to
    the handful of bad channels it is looking for -- a plain mean and standard deviation
    would be dragged by the very outliers being detected.

    This runs *before* re-referencing on purpose. Average referencing mixes every
    channel into every other, so one dead electrode left in the average contaminates
    the whole montage.
    """
    picks = [
        i for i in mne.pick_types(raw.info, eeg=True)
        if raw.ch_names[i] != REFERENCE_CHANNEL
    ]
    data = raw.get_data(picks=picks)
    log_var = np.log(data.var(axis=1) + 1e-30)
    median = np.median(log_var)
    mad = np.median(np.abs(log_var - median))
    if mad == 0:
        return []
    z = 0.6745 * (log_var - median) / mad
    names = [raw.ch_names[i] for i in picks]
    return [n for n, zi in zip(names, z) if abs(zi) > z_threshold]


def preprocess(
    raw: mne.io.BaseRaw,
    remove_ocular: bool = True,
    random_state: int = 0,
) -> mne.io.BaseRaw:
    """Filter, repair bad channels, re-reference, and optionally remove ocular components.

    Order matters and is not arbitrary:

    1. Band-pass first, so that drift does not distort the variance statistics used for
       bad-channel detection.
    2. Detect and interpolate bad channels *before* re-referencing (see find_bad_channels).
    3. Average reference. The recording reference (Cz) sits over central cortex, so with
       it in place every channel carries an inverted copy of central activity and scalp
       topographies cannot be read. Average reference is a linear remix, so it barely
       changes what a model can learn -- it changes what a *human* can interpret.
    4. ICA last, on a copy high-passed at 1 Hz, with blink components identified by
       correlation against the EOG electrodes rather than by eye.
    """
    raw = raw.copy().filter(HIGHPASS_HZ, LOWPASS_HZ, fir_design="firwin", verbose="ERROR")
    raw.resample(TARGET_SFREQ, verbose="ERROR")

    raw.info["bads"] = find_bad_channels(raw)
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")

    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    if remove_ocular:
        ica = mne.preprocessing.ICA(
            n_components=ICA_N_COMPONENTS,
            method="fastica",
            random_state=random_state,
            max_iter="auto",
        )
        ica.fit(
            raw.copy().filter(ICA_FIT_HIGHPASS_HZ, None, fir_design="firwin", verbose="ERROR"),
            decim=5,
            verbose="ERROR",
        )
        bad_ica, _ = ica.find_bads_eog(raw, verbose="ERROR")
        ica.exclude = bad_ica
        ica.apply(raw, verbose="ERROR")
        raw.info["temp"] = {"n_ocular_components_removed": len(bad_ica)}

    return raw


def make_epochs(
    raw: mne.io.BaseRaw,
    trials: pd.DataFrame,
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
    baseline: tuple[float, float] = BASELINE,
) -> mne.Epochs:
    """Cut one epoch per studied word, tagged by recall outcome.

    ``trials`` must be the rows of the trial table for this subject and session.

    The window is wider than the ERP question needs (-1 to +2 s rather than -0.2 to
    +1.6). Time-frequency analysis in a later session convolves wavelets that require
    signal either side of the interval of interest; extracting narrow now would mean
    re-reading every EDF later. Words are ~3.9 s apart, so these epochs do not overlap.

    Baseline is -200 to 0 ms: late enough that the previous word's response has passed,
    early enough that visual information from this word has not yet reached cortex.
    Whatever sits in the baseline window is subtracted into the rest of the epoch, so it
    must contain no response to the word being measured.
    """
    onsets = trials["onset_s"].to_numpy()
    samples = np.round(onsets * raw.info["sfreq"]).astype(int) + raw.first_samp
    codes = trials["recalled"].to_numpy().astype(int)
    events = np.column_stack([samples, np.zeros_like(samples), codes])

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"forgotten": 0, "recalled": 1},
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject_by_annotation=False,
        verbose="ERROR",
    )
    epochs.metadata = trials.reset_index(drop=True)
    return epochs


#: AutoReject parallelises through joblib, and each worker holds a copy of the epochs
#: array (~190 MB). On a 16 GB machine, 8 workers plus the parent is enough to reach swap
#: when anything else is running; a worker then gets killed and the pool is poisoned for
#: the rest of the process. Four is comfortable and barely slower, since the work is
#: memory-bandwidth-bound rather than core-bound.
AUTOREJECT_N_JOBS = 2


def repair_epochs(
    epochs: mne.Epochs, n_jobs: int = AUTOREJECT_N_JOBS, random_state: int = 0
) -> mne.Epochs:
    """Interpolate bad channels per epoch and drop the unsalvageable ones.

    AutoReject is unsupervised -- it sees amplitudes, never the recall label -- so it
    cannot leak the outcome being predicted. It is also fitted per session, and the
    evaluation protocol holds out whole sessions, so a fitted threshold never crosses
    the train/test boundary.
    """
    ar = AutoReject(
        n_interpolate=AUTOREJECT_N_INTERPOLATE,
        consensus=AUTOREJECT_CONSENSUS,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=False,
    )
    return ar.fit_transform(epochs, return_log=False)


def epochs_path(subject: str, session: int, remove_ocular: bool) -> Path:
    tag = "ocular-removed" if remove_ocular else "ocular-kept"
    return DERIV_ROOT / subject / f"{subject}_ses-{session}_{tag}_epo.fif"


def build_session_epochs(
    subject: str,
    session: int,
    trials: pd.DataFrame,
    remove_ocular: bool = True,
    cache: bool = True,
    repair: bool = True,
    n_jobs: int = AUTOREJECT_N_JOBS,
) -> mne.Epochs:
    """Full pipeline for one session, cached to disk as FIF.

    Preprocessing a session takes minutes; the cache makes every later analysis instant.
    Corrected and uncorrected runs are cached separately so the two can be compared
    without recomputation.
    """
    out = epochs_path(subject, session, remove_ocular)
    if cache and out.exists():
        return mne.read_epochs(out, preload=True, verbose="ERROR")

    raw = preprocess(load_raw(subject, session), remove_ocular=remove_ocular)
    session_trials = trials[
        (trials["subject_id"] == subject) & (trials["session"] == session)
    ].sort_values("onset_s")
    epochs = make_epochs(raw, session_trials)
    if repair:
        epochs = repair_epochs(epochs.pick("eeg"), n_jobs=n_jobs)

    if cache:
        out.parent.mkdir(parents=True, exist_ok=True)
        epochs.save(out, overwrite=True, verbose="ERROR")
    return epochs
