import os
import mne
import librosa
import numpy as np
import pandas as pd
import pyorganoid as po
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *
from pathlib import Path
from scipy.signal import welch
__import__("warnings").filterwarnings("ignore")


data_dir = Path('data')


def preprocess_and_epoch(subject_id):
    raw = load_eeg_data(data_dir, subject_id)

    # Apply band-pass filter
    raw.filter(1., 40., fir_design='firwin')

    # Fit and apply ICA for artifact correction
    ica = mne.preprocessing.ICA(n_components=20, random_state=97)
    ica.fit(raw)
    raw = ica.apply(raw)

    # Load events and create epochs
    events_df = load_eeg_events(data_dir, subject_id)
    events_df['onset_samples'] = (events_df['onset'] * raw.info['sfreq']).astype(int)
    events_array = np.column_stack((events_df['onset_samples'],
                                    np.zeros(len(events_df), dtype=int), events_df['trial_type']))
    event_id = {str(evt): evt for evt in np.unique(events_df['trial_type'])}
    epochs = mne.Epochs(raw, events_array, event_id=event_id, tmin=-0.2, tmax=10.0, baseline=(None, 0),
                        preload=True, event_repeated='merge')

    # Save epoch temporarily
    epochs.save(data_dir / 'epochs' / f'sub-{subject_id}_epochs-epo.fif', overwrite=True)


def process_all_subjects():
    for subject_id in range(1, 22):
        print("Preprocessing and epoching subject", subject_id)
        preprocess_and_epoch(subject_id)
    print("Finished preprocessing and epoching all subjects. Combining epochs...")


def combine_epochs():
    # epochs_files = list(data_dir.glob('epochs/sub-*_epochs-epo.fif'))
    # all_epochs = [mne.read_epochs(f) for f in epochs_files]
    # combined_epochs = mne.concatenate_epochs(all_epochs)
    # combined_epochs.save(data_dir / 'epochs' / 'combined_epochs-epo.fif', overwrite=True)

    # Combine epochs for subjects 1-5 only (because each subject's epoch data is ~3-5 GB in total)
    epochs_files = list(data_dir.glob('epochs/sub-[1-5]_epochs-epo.fif'))
    all_epochs = [mne.read_epochs(f) for f in epochs_files]
    combined_epochs = mne.concatenate_epochs(all_epochs)
    combined_epochs.save(data_dir / 'epochs' / 'combined_epochs-1to5-epo.fif', overwrite=True)
    print("Finished combining epochs.")


def train_model():
    pass


if __name__ == "__main__":
    print("Hello, world!")
    # process_all_subjects()
    combine_epochs()
    # train_model()
