import os
import mne
import librosa
import numpy as np
import pandas as pd
import pyorganoid as po
import tensorflow as tf
import concurrent.futures
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
    return epochs


def combine_epochs():
    all_epochs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(preprocess_and_epoch, subject_id) for subject_id in range(1, 22)]
        for future in concurrent.futures.as_completed(futures):
            all_epochs.append(future.result())
    combined_epochs = mne.concatenate_epochs(all_epochs)
    combined_epochs.save(data_dir / 'epochs' / 'combined_epochs-epo.fif', overwrite=True)


def train_model():
    pass


if __name__ == "__main__":
    print("Hello, world!")
    # train_model()
    combine_epochs()
