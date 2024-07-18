import os
import mne
import sys
import librosa
import logging
import warnings
import numpy as np
import pandas as pd
import pyorganoid as po
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *
from pathlib import Path
from scipy.signal import welch
from tensorflow.keras import layers, models, optimizers


mne.set_log_level('WARNING')
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
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
    """Deprecated due to system memory constraints."""
    # epochs_files = list(data_dir.glob('epochs/sub-*_epochs-epo.fif'))
    # all_epochs = [mne.read_epochs(f) for f in epochs_files]
    # combined_epochs = mne.concatenate_epochs(all_epochs)
    # combined_epochs.save(data_dir / 'epochs' / 'combined_epochs-epo.fif', overwrite=True)
    # Combine epochs for subjects 1-5 only (because each subject's epoch data is ~3-5 GB in total)
    # epochs_files = list(data_dir.glob('epochs/sub-[1-5]_epochs-epo.fif'))
    # all_epochs = [mne.read_epochs(f) for f in epochs_files]
    # combined_epochs = mne.concatenate_epochs(all_epochs)
    # combined_epochs.save(data_dir / 'epochs' / 'combined_epochs-1to5-epo.fif', overwrite=True)
    # print("Finished combining epochs.")
    pass


def extract_audio_features():
    audio_features = {}
    audio_files = list(data_dir.glob('mp3/*.mp3'))
    for i, file_name in enumerate(audio_files):
        y, sr = librosa.load(file_name, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
        # mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / np.std(mfccs, axis=1, keepdims=True)  # Normalize
        audio_features[i+1] = mfcc.T  # Transpose to align time steps as rows
    return audio_features


def load_eeg_data_memmapped(file_path):
    # Example Usage: eeg_data, times = load_eeg_data_memmapped(Path(f"data/epochs/sub-{subject_id}_epochs-epo.fif"))
    epochs = mne.read_epochs(file_path, preload=False, verbose=False)  # Preload=False to avoid loading data into memory
    memmap_file = f"data/epochs/memmap/{file_path.stem}_memmapped.npy"  # Create a .npy filename for memmapped data
    # Load memmap if it exists, otherwise create it
    if os.path.exists(memmap_file):
        data = np.memmap(memmap_file, dtype='float32', mode='r+', shape=epochs.get_data().shape)
    else:
        data = np.memmap(memmap_file, dtype='float32', mode='w+', shape=epochs.get_data().shape)
        data[:] = epochs.get_data().astype('float32')  # Load data into memmapped array
    return data, epochs.times


def data_generator(eeg_files, audio_features, batch_size=32, verbose=False):
    while True:  # Infinite loop for keras fit_generator
        for eeg_file in eeg_files:
            subject_id = int(eeg_file.stem.split('_')[0].split('-')[1])
            try:
                eeg_data, _ = load_eeg_data_memmapped(eeg_file)
                audio_data = audio_features.get(subject_id)

                if audio_data is None:
                    if verbose:
                        print(f"Skipping {eeg_file} due to missing audio features.")
                    continue

                indices = np.arange(len(eeg_data))
                np.random.shuffle(indices)

                for start_idx in range(0, len(eeg_data), batch_size):
                    end_idx = min(start_idx + batch_size, len(eeg_data))
                    batch_indices = indices[start_idx:end_idx]

                    # Ensure that the batch indices do not exceed the audio data length
                    if len(audio_data) < len(batch_indices):
                        if verbose:
                            print(f"Audio data for subject {subject_id} is shorter than the batch size.")
                        continue

                    yield eeg_data[batch_indices], audio_data[batch_indices]
            except KeyError as e:
                if verbose:
                    print(f"Skipping {eeg_file} due to KeyError: {e}")
            except Exception as e:
                if verbose:
                    print(f"An error occurred while processing {eeg_file}: {e}")


def build_model(input_shape, units=64, dropout_rate=0.5, recurrent_dropout=0.5, num_classes=1,
                final_activation='linear', loss=None, metrics=None):
    if final_activation is None:
        final_activation = 'sigmoid' if num_classes == 1 else 'softmax'
    if loss is None:
        loss = 'binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy'
    if metrics is None:
        metrics = ['accuracy']
    model = models.Sequential([
        layers.LSTM(units, input_shape=input_shape, recurrent_dropout=recurrent_dropout, return_sequences=True),
        # layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        # layers.LSTM(units, recurrent_dropout=recurrent_dropout),
        layers.LSTM(units, return_sequences=False),
        # layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(units, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation=final_activation)
    ])
    optimizer = optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def train_model():
    # Load a sample epoch to determine input shape for the model
    # This shape of the data typically returns (n_epochs, n_channels, n_times)
    # For input shape to LSTM, we care about (n_channels, n_times) for a single epoch
    sample_epochs = mne.read_epochs(data_dir/'epochs'/'sub-1_epochs-epo.fif', preload=True)
    sample_data = sample_epochs.get_data(copy=False)
    input_shape = sample_data.shape[1:]

    audio_features = extract_audio_features()
    eeg_files = [Path(f"data/epochs/sub-{i}_epochs-epo.fif") for i in range(1, 22)]
    train_gen = data_generator(eeg_files, audio_features)

    model = build_model(input_shape, units=128, dropout_rate=0.5, loss='mse', metrics=['mae'],
                        recurrent_dropout=0.25, num_classes=1)  # 1 class for continuous valence
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, to_file='images/lstm_model.png')

    history = model.fit(train_gen, steps_per_epoch=100, epochs=10, verbose=1)
    model.save('weights/lstm_model.h5')

    for key in history.history:
        plt.plot(history.history[key])
        plt.title(f'Model {key.title()}')
        plt.ylabel(key.title())
        plt.xlabel('Epoch')
        plt.show()
    pass


if __name__ == "__main__":
    print("Hello, world!")
    # process_all_subjects()
    # combine_epochs()
    train_model()
