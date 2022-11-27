import os
import mne
from mne import viz
from preprocessing import Pipeline

"""This file takes as input a clean EEG segment and outputs a resampled, filtered signal which does not contain 
bad channels(as per PREP pipeline). It also performs ICA on the given segment, and finally performs current source 
density on the raw data. It outputs the ready to be used EEG signals in raw format."""

p = Pipeline("C:/Srirupa/Mitacs Project/Clean EEG/processed_data_1.edf")
p.applyPipeline(500, 12)

raw = p.getRaw()
raw.drop_channels(raw.info['bads'])

raw_csd = mne.preprocessing.compute_current_source_density(raw)
artifact_picks_1 = mne.pick_channels(raw_csd.ch_names, include=raw_csd.ch_names)
raw_csd.plot(order=artifact_picks_1, n_channels=len(artifact_picks_1),duration=0.5, show_scrollbars=False, block = True)
