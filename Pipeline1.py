import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from edf_extraction import Extractor

"""This file takes as input a raw EEG signal in EDF format and outputs segments of clean EEG signal 
from the raw signal. It can produce multiple clean EEG signal for a single raw EEG signal given as
input. The given code identifies and removes intervals of special procedures performed on patients 
during recordings, such as hyperventilation (deep breathing) and photic stimulation (flashing light). 
Physicians apply these tests to patients in order to detect brain abnormal activity for epilepsy 
diagnosis. Since these procedures burst abnormal activity, and weren't performed for all subjects, 
we exclude them from the analysis. Also the recordings contain intervals with no signal. It is 
the result of turned off equipment or disconnected electrode. So we also have to avoid these flat 
intervals with zero signal. Thus traget slices acquired only from clean intervals from each EEG, 
without flat intervals, hyperventilation and photic stimulation. Slices taken from the beginning, 
first minute taken as "bad" by default. The algoritm also handles cases when "bad intervals" overlap"""

filepath = "C:/Srirupa/EEG Prepocessing/Raw EEG/sample_data.edf"
output_path = "C:/Srirupa/EEG Prepocessing/Clean EEG"

# Initiate the preprocessing(cleaning) object, filter the data between 0.5 Hz and 55 Hz and resample to 500 Hz.
p = Extractor(filepath, target_frequency=500)

# This calls internal functions to detect 'bad intervals' and define 5 'good' ones 60 seconds each
p.extract_good(target_length=60, target_segments=5)

# Calling the function saves new clean EDF files to output_folder. In case there are more than 1, it adds suffix "_n" to the file name 
p.save_edf(folder=output_path, filename='clean_data.edf')


