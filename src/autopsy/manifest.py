from pathlib import Path
import pandas as pd

def build_manifest(dataset_root: Path) -> pd.DataFrame:
    """Scan a BIDS EEG dataset directory and summarise each recording.

    Returns one row per recording, sorted by subject, with columns:
        subject          str    e.g. "sub-001"
        task             str    e.g. "activeai"
        sfreq            float  from *_eeg.json "SamplingFrequency"
        n_eeg_channels   int    from *_eeg.json "EEGChannelCount"
        duration_s       float  from *_eeg.json "RecordingDuration"
        n_events         int    data rows in *_events.tsv (excludes the header)
    """


list