# About
Pianoid is a deep Organoid Learning system (comprised of a Bidirectional LSTM Network and EEG-Response Organoid) trained to provide realistic EEG signal responses to classical music.
These signal responses are interfaced through a simulated organoid (using the novel [pyorganoid library](https://github.com/danielathome19/pyorganoid)) to mimic a human brain response.
This project was designed to demonstrate the capabilities of `pyorganoid` and its usage in studying the new field of Organoid Intelligence.

To find out more, check out the provided research paper:
  * "Simulation of Neural Responses to Classical Music Using Organoid Intelligence Methods" (DOI: [TBD](https://doi.org/TBD))

# Usage
For data used in my experiments:
  * All datasets can be found in **data**.
  * My most recent pre-trained weights can be found on [Hugging Face](https://huggingface.co/danielathome19/Pianoid-EEG-NN/blob/main/lstm_model.h5). These should be stored in a folder named **weights**.

**NOTE:** these folders should be placed in the **same** folder as "main.py". For folder existing conflicts, simply merge the directories.

In main.py, the "main" function acts as the controller for the model, where calls to train the model, create a prediction, and all other functions are called. One may also call these functions from an external script (`from main import simulate_organoid`, etc.).

To choose an operation or series of operations for the model to perform, simply edit the main function before running. Examples of all function calls can be seen commented out within main.

# Bugs/Features
Bugs are tracked using the GitHub Issue Tracker.

Please use the issue tracker for the following purpose:
  * To raise a bug request; do include specific details and label it appropriately.
  * To suggest any improvements in existing features.
  * To suggest new features or structures or applications.
  
# License
The code is licensed under CC0 License 1.0.

The dataset was compiled from free and open sources with respect to the original file creators and sequencers. This work is purely for educational and research purposes, and no copyright is claimed on any files contained within the dataset.
