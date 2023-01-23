from edf_extraction import slice_edfs
from multiple_preprocessing import get_processed_data
import pandas as pd
import mne
import warnings
warnings.filterwarnings('ignore')
mne.set_log_level('warning')

source_folder = "C:/Srirupa/EEG Prepocessing/Raw EEG"
intermediate_folder = "C:/Srirupa/EEG Prepocessing/Clean EEG"
target_folder = "C:/Srirupa/EEG Prepocessing/Processed EEG"

# Looks for raw EEG files in source folder, and apply the preprocessing(cleaning) to 100 files in total
# resample to 500 Hz, extract 5 segment of 60 seconds from each EDF file
# saves new segments as EDF into intermediate folder
slice_edfs(source_folder=source_folder, target_folder=intermediate_folder, 
           target_frequency=500, target_length=60, target_segments=5, nfiles=100)
           
# if you don't need files limit - don't specify the parameter "nfiles", default is None


# Looks for clean EEG files in the intermediate folder, and apply the preprocessing to 100 files in total
# resample to 500 Hz, filters signal which does not contain bad channels(as per PREP pipeline). 
# It also performs ICA on the given segment and outputs the ready pre-processed EEG signals in raw format 
# and saves new segments as EDF into target folder
get_processed_data(source_folder=intermediate_folder, target_folder=target_folder, 
            target_frequency=500, n_components=12, nfiles=100)
