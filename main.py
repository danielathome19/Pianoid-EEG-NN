import gc
import os
import mne
import sys
import pickle
import librosa
import logging
import warnings
import numpy as np
import pandas as pd
import pyorganoid as po
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *
from organoid import *
from pathlib import Path
from scipy.signal import welch
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers


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
    print("Finished preprocessing and epoching all subjects.")


def extract_audio_features(file=None):
    if file is not None:
        # Extract audio features for a single file
        y, sr = librosa.load(file, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
        return mfcc.T
    audio_features = {}
    audio_files = list(data_dir.glob('mp3/*.mp3'))
    for i, file_name in enumerate(audio_files):
        y, sr = librosa.load(file_name, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=512)
        # mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)  # Normalize
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


def normalize_data(data, scaler):
    # Assume data shape is (batch_size, channels, features)
    # Reshape data to (batch_size * channels, features) for scaling
    shaped_data = data.reshape(-1, data.shape[-1])
    return scaler.transform(shaped_data).reshape(data.shape)


def data_generator(eeg_files, audio_features, audio_input_shape, batch_size=32, verbose=False, scaler=None):
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
                if scaler:
                    eeg_data = normalize_data(eeg_data, scaler)
                indices = np.arange(len(eeg_data))
                np.random.shuffle(indices)
                for start_idx in range(0, len(eeg_data), batch_size):
                    end_idx = min(start_idx + batch_size, len(eeg_data))
                    batch_indices = indices[start_idx:end_idx]
                    # Ensure the batch indices do not exceed the audio data length
                    if len(audio_data) < len(batch_indices):
                        if verbose:
                            print(f"Audio data for subject {subject_id} is shorter than the batch size.")
                        continue
                    eeg_batch = eeg_data[batch_indices]
                    audio_batch = audio_data[:len(batch_indices)]
                    if audio_batch.shape[0] < eeg_batch.shape[0]:
                        audio_batch = np.pad(audio_batch, ((0, eeg_batch.shape[0] - audio_batch.shape[0]), (0, 0)), 'constant')
                    if audio_batch.ndim == 2:
                        audio_batch = np.expand_dims(audio_batch, axis=-1)
                    audio_batch = np.tile(audio_batch, (1, 1, audio_input_shape[1]))
                    yield audio_batch, eeg_batch
            except KeyError as e:
                if verbose:
                    print(f"Skipping {eeg_file} due to KeyError: {e}")
            except Exception as e:
                if verbose:
                    print(f"An error occurred while processing {eeg_file}: {e}")
    pass


def fit_scaler():
    print("Fitting the StandardScaler...")
    scaler = StandardScaler()
    epochs_files = [Path(f"data/epochs/sub-{i}_epochs-epo.fif") for i in range(1, 22)]
    for file_path in epochs_files:
        print(f"Fitting scaler for {file_path}")
        data, _ = load_eeg_data_memmapped(file_path)
        # Reshape data to 2D (samples, features) if necessary
        reshaped_data = data.reshape(-1, data.shape[-1])
        scaler.partial_fit(reshaped_data)
    with open('weights/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    pass


def build_model(audio_input_shape, eeg_output_shape, units=64, dropout_rate=0.5, recurrent_dropout=0.5,
                loss='mse', metrics=('mae',), l2_reg=0.01):
    model = models.Sequential([
        layers.Input(shape=audio_input_shape, name='audio_input'),
        layers.Bidirectional(layers.LSTM(units, dropout=dropout_rate, recurrent_dropout=recurrent_dropout,
                                         return_sequences=True)),
        layers.Bidirectional(layers.LSTM(units, dropout=dropout_rate, recurrent_dropout=recurrent_dropout)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.Dense(np.prod(eeg_output_shape), activation='linear'),
        layers.Reshape(eeg_output_shape)
    ])

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    checkpoint = callbacks.ModelCheckpoint('weights/lstm_model_best.h5', monitor='val_loss', save_best_only=True)
    lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.9)
    optimizer = optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model, [early_stopping, reduce_lr, checkpoint]


def train_model():
    print("Training the model...")
    # Load a sample epoch to determine input shape for the model
    # This shape of the data typically returns (n_epochs, n_channels, n_times)
    # For input shape to LSTM, we care about (n_channels, n_times) for a single epoch
    if not os.path.exists('weights/eeg_output_shape.pkl'):
        sample_epochs = mne.read_epochs(data_dir/'epochs'/'sub-1_epochs-epo.fif', preload=True)
        sample_data = sample_epochs.get_data(copy=False)
        eeg_output_shape = sample_data.shape[1:]  # (n_channels, n_times)
        with open('weights/eeg_output_shape.pkl', 'wb') as f:
            pickle.dump(eeg_output_shape, f)
    else:
        with open('weights/eeg_output_shape.pkl', 'rb') as f:
            eeg_output_shape = pickle.load(f)

    audio_features = extract_audio_features()
    audio_input_shape = (audio_features[1].shape[0], audio_features[1].shape[1])
    eeg_files = [Path(f"data/epochs/sub-{i}_epochs-epo.fif") for i in range(1, 22)]
    train_files, val_files = train_test_split(eeg_files, test_size=0.15, random_state=42)

    batch_size = 32
    train_gen = data_generator(train_files, audio_features, audio_input_shape, batch_size=batch_size)
    val_gen = data_generator(val_files, audio_features, audio_input_shape, batch_size=batch_size)

    model, m_callbacks = build_model(audio_input_shape, eeg_output_shape, units=32,
                                     dropout_rate=0.1, recurrent_dropout=0.1)
    model.build(input_shape=(None, ) + audio_input_shape)
    model.summary()
    plot_model(model, show_shapes=True, expand_nested=True,
               to_file='images/lstm_model.png', show_layer_activations=True, dpi=300)

    gc.disable()  # Disable garbage collection to prevent memory issues and stalling during training
    history = model.fit(train_gen, steps_per_epoch=100, epochs=25, verbose=1,
                        callbacks=m_callbacks, validation_data=val_gen, validation_steps=25)
    model.save('weights/lstm_model.h5')
    gc.enable()

    for key in ['loss', 'mae']:
        key = key.title() if key == 'loss' else key.upper()
        plt.figure(figsize=(12, 6))
        plt.plot(history.history[key.lower()], label=f'Training {key}')
        plt.plot(history.history[f'val_{key.lower()}'], label=f'Validation {key}')
        plt.legend()
        plt.title(f'Model {key}')
        plt.ylabel(key)
        plt.xlabel('Epoch')
        plt.savefig(f'images/model_{key.lower()}_history.png')
        plt.show()
    pass


def test_model():
    print("Testing the model...")
    model = models.load_model('weights/lstm_model.h5')
    mp3_file = data_dir/'mp3/p1_chopin-n10-op12-bertoglio.mp3'
    mp3_features = extract_audio_features(file=mp3_file)
    mp3_features = np.expand_dims(mp3_features, axis=-1) if mp3_features.ndim == 2 else mp3_features  # Ensure 3D shape
    predicted_eeg = model.predict(np.expand_dims(mp3_features, axis=0))  # Predict EEG activities

    # Load true EEG data for comparison
    true_eeg = mne.read_epochs(data_dir/'epochs/sub-1_epochs-epo.fif', preload=True).get_data(copy=False)

    # Reshape and trim true_eeg to match the predicted shape (batch_size, n_channels, n_times)
    true_eeg = true_eeg[:, :, :predicted_eeg.shape[2]]  # Ensure it matches the time steps
    true_eeg = true_eeg.mean(axis=0, keepdims=True)  # Compute the average across epochs

    # Calculate the channel-wise correlation between predicted and true EEG
    corr = np.corrcoef(predicted_eeg.flatten(), true_eeg.flatten())[0, 1]
    print(f"Correlation between predicted and true EEG: {corr:.4f}")  # rho=0.5073, future work: augment dataset

    scaler = StandardScaler()
    scaled_predicted_eeg_norm = (scaler.fit_transform(predicted_eeg[0].reshape(-1, predicted_eeg.shape[2]))
                                 .reshape(predicted_eeg[0].shape))
    scaled_true_eeg_norm = scaler.fit_transform(true_eeg[0].reshape(-1, true_eeg.shape[2])).reshape(true_eeg[0].shape)

    # Calculate the power spectral density of the predicted and true EEG
    f, pxx_predicted = welch(scaled_predicted_eeg_norm.flatten(), fs=1000, nperseg=256)
    f, pxx_true = welch(scaled_true_eeg_norm.flatten(), fs=1000, nperseg=256)

    # Plot the power spectral density of the predicted and true EEG
    plt.figure(figsize=(12, 6))
    plt.plot(f, pxx_true, label='True EEG')
    plt.plot(f, pxx_predicted, label='Predicted EEG')
    plt.legend()
    plt.title('Power Spectral Density of Normalized Predicted vs. True EEG Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.savefig('images/psd_predicted_vs_true_eeg.png')
    plt.show()

    # Calculate the average EEG signal across all channels
    avg_predicted_eeg = np.mean(predicted_eeg[0], axis=0)
    avg_true_eeg = np.mean(true_eeg[0], axis=0)

    # Trim values to be between 0 and 0.5 for better visualization
    avg_predicted_eeg = np.clip(avg_predicted_eeg, -0.01, 0.01)
    avg_true_eeg = np.clip(avg_true_eeg, -0.01, 0.01)

    # Plot the average predicted EEG vs. true EEG
    plt.figure(figsize=(12, 6))
    plt.plot(avg_true_eeg, label='True EEG')
    plt.plot(avg_predicted_eeg, label='Predicted EEG')
    plt.legend()
    plt.title('Average Actual vs. Predicted EEG Signal')
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.savefig('images/average_actual_vs_predicted_eeg.png')
    plt.show()
    print("Testing completed and plots saved.")


def simulate_organoid(mp3_filepath=None):
    print("Simulating the organoid...")
    model_path = "weights/lstm_model.h5"
    ml_model = po.TFModel(model_path, input_shape=(None, 20, 1))

    environment = AudioEnvironment()
    organoid = EEGOrganoid(environment, ml_model, num_cells=47)  # Corresponding to 47 EEG channels from the dataset
    organoid.plot_organoid("images/eeg_organoid.png", show_properties=True, dpi=300, truncate_cells=6)
    mp3_file = mp3_filepath or data_dir/'mp3/p1_chopin-n10-op12-bertoglio.mp3'
    simulator = AudioScheduler(organoid)
    simulator.simulate(mp3_file)

    organoid.plot_simulation_history('Frequency Over Time', 'Frequency',
                                     filename='images/eeg_organoid_simulation.png', dpi=300)
    
    # Create EEG frequency plot
    n_cells = len(organoid.get_cells())
    fig, axes = plt.subplots(n_cells, 1, figsize=(12, 36), sharex=True)
    for i, cell in enumerate(organoid.get_cells()):
        axes[i].plot(cell.get_history(), color='blue')
        axes[i].set_ylabel(f'Cell {i+1}', rotation=0, labelpad=20, ha='right')
        axes[i].set_yticks([])  # Remove frequency labels
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
    axes[-1].set_xlabel('Time Steps', fontsize=14, labelpad=10)
    plt.suptitle('Organoid Cell Activity Over Time', fontsize=16)
    plt.savefig('images/eeg_organoid_frequency_over_time.png', dpi=300)


if __name__ == "__main__":
    print("Hello, world!")
    # process_all_subjects()
    # combine_epochs()
    # fit_scaler()
    # train_model()
    # test_model()
    simulate_organoid()
