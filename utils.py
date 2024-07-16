import mne
import librosa
import numpy as np
import pandas as pd
from pathlib import Path


def split_edf(file_path, output_dir, part_duration_sec):
    """
    Split an EDF file into multiple smaller segments.

    Parameters
    ----------
    file_path : Path
        Path to the original EDF file.
    output_dir : Path
        Directory to save the split parts.
    part_duration_sec : int
        Duration of each part in seconds.
    """
    raw = mne.io.read_raw_edf(file_path, preload=True)
    total_duration_sec = int(raw.times[-1])
    n_parts = total_duration_sec // part_duration_sec + (1 if total_duration_sec % part_duration_sec != 0 else 0)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_parts):
        start_sec = i * part_duration_sec
        stop_sec = start_sec + part_duration_sec
        if stop_sec > total_duration_sec:
            stop_sec = total_duration_sec
        part_raw = raw.copy().crop(tmin=start_sec, tmax=stop_sec, include_tmax=False)
        part_filename = output_dir / f"{file_path.stem}_part{i}.fif"
        part_raw.save(part_filename, overwrite=True)


def load_edf_from_parts(directory, pattern):
    """
    Load and concatenate EDF parts from a directory.

    Parameters
    ----------
    directory : Path
        Directory containing the split EDF parts.
    pattern : str
        Pattern to match the files (e.g., 'sub-01*part*.fif').

    Returns
    -------
    mne.io.Raw
        The concatenated Raw object.
    """
    parts = list(directory.glob(pattern))
    raw_parts = [mne.io.read_raw_fif(p, preload=True) for p in sorted(parts)]
    raw_concat = mne.concatenate_raws(raw_parts)
    # Convert data from float64 to float32 to save memory
    raw_concat.pick_types(eeg=True, meg=False, stim=True)  # Ensure only EEG and STIM channels are loaded if necessary
    raw_concat.load_data()
    raw_concat.apply_function(lambda x: x.astype(np.float32))
    return raw_concat


def extract_audio_features(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Normalize MFCCs
    mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / np.std(mfccs, axis=1, keepdims=True)
    return mfccs.T  # Transpose to have time steps as rows for LSTM


def load_eeg_data(data_dir, subject_id):
    return load_edf_from_parts(data_dir/'eeg'/f'sub-{subject_id:02d}', f'sub-{subject_id:02d}*part*.fif')
    # return mne.io.read_raw_edf(data_dir/'eeg'/ f'sub-{subject_id:02d}_task-classicalMusic_eeg.edf', preload=True)


def load_eeg_events(data_dir, subject_id):
    return pd.read_csv(data_dir/'eeg_events'/f'sub-{subject_id:02d}_task-classicalMusic_events.tsv', sep="\t")


if __name__ == "__main__":
    for j in range(1, 22):
        split_edf(Path(f"data/eeg/sub-{j:02d}_task-classicalMusic_eeg.edf"), Path(f"data/eeg/sub-{j:02d}"), 300)
