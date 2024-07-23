import numpy as np
import pyorganoid as po
from main import extract_audio_features
from pyorganoid import generate_random_position as gen_rand_pos


class AudioEnvironment(po.Environment):
    def __init__(self, dimensions=3, size=100.0):
        super().__init__(dimensions, size)
        self.audio_features = None

    def load_mp3(self, mp3_file):
        self.audio_features = extract_audio_features(mp3_file)


class EEGCell(po.Cell):
    def __init__(self, position, channel_index):
        super().__init__(position, input_data_func=(lambda _: None))
        self.channel_index = channel_index


class EEGModule(po.BaseMLModule):
    def run(self, agent, verbose=False):
        pass


class EEGOrganoid(po.Organoid):
    def __init__(self, environment, ml_model, num_cells=47):
        super().__init__(environment)
        for i in range(num_cells):
            cell = EEGCell(position=tuple(gen_rand_pos(environment.dimensions, environment.size)), channel_index=i)
            cell.add_module(EEGModule(ml_model))
            self.add_agent(cell)
        pass


class AudioScheduler(po.Scheduler):
    def simulate(self, mp3_file):
        # Load and process the MP3 file in the environment
        self.organoid.environment.load_mp3(mp3_file)
        audio_features = self.organoid.environment.audio_features
        num_timesteps = audio_features.shape[0]

        # Predict EEG signals using the model
        input_data = np.expand_dims(audio_features, axis=0)
        predicted_eeg = self.organoid.agents[0].modules[0].ml_model.model.predict(input_data)

        # Update each cell with its corresponding EEG channel data for all time steps
        for timestep in range(num_timesteps):
            print(f"Step {timestep+1}/{num_timesteps}")
            for i, agent in enumerate(self.organoid.agents):
                agent.update(predicted_eeg[0, i, timestep])
        pass
