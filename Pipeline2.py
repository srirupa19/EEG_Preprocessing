import os
import mne
from mne import viz
from individual_func import write_mne_edf
from preprocessing import Pipeline

"""This file takes as input a clean EEG segment and outputs a resampled, filtered signal which does not contain 
bad channels(as per PREP pipeline). It also performs ICA on the given segment, and finally performs current source 
density on the raw data. It outputs the ready to be used EEG signals in raw format."""

filepath = "C:/Srirupa/EEG Prepocessing/Clean EEG/clean_data_1.edf"
output_path = "C:/Srirupa/EEG Prepocessing/Processed EEG/processed_data_1.edf"

# Initiate the preprocessing object
p = Pipeline(filepath)

# Calling the function filters the data between 0.5 Hz and 55 Hz, resamples to 500 Hz
# and performs ICA after applying the PREP pipeline to remove bad channels
p.applyPipeline(500, 12)

# Calling the function gets the pre-processed data in raw format
raw = p.getRaw()

# Calling the function drops the bad channels(as per PREP pipeline)
raw.drop_channels(raw.info['bads'])

# Calling the function saves pre-processed EDF files to output_folder.
write_mne_edf(raw, fname=output_path, overwrite=True)

# Applies current source density on raw EEG files
raw_csd = mne.preprocessing.compute_current_source_density(raw)

# Plots the current source density of the pre-proceesed EEG
artifact_picks_1 = mne.pick_channels(raw_csd.ch_names, include=raw_csd.ch_names)
raw_csd.plot(order=artifact_picks_1, n_channels=len(artifact_picks_1),duration=0.5, show_scrollbars=False, block = True)
