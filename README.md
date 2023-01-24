# EEG preprocessing pipeline for Clinical EEG dataset

## Summary

The code is aimed to preprocess clinical EEG recordings and make them a suitable input for later analysis and ML applications. 
- **Input**: EDF file
- **Output**: 
  - (1) new EDF file(s) containing only clean and pre-processed EEG interval of target length
- There are three types of EDF files. 
  - (1) Raw EEG folder - Containing the raw clinical EEG files
  - (2) Clean EEG folder - Contains the clean EEG segments after applying Pipeline1
  - (3) Processed EEG folder - Contains the pre-processed EEG segments to be used for classification and other purposes

**Parameters**:
- number of clean EEG slices to extract from each EDF file
- target length of each slice, in seconds
- targent frequency for resampling the signal
- number of components to be used in performing ICA

## Performed operations

### Pipeline 1 - Nikolay's Code
1) resample each recording's signal to 500 Hz frequency, since some recordings might have different sampling frequencies. 
2) applied frequency filtering that keeps only signal with frequencies between 0.5 and 55 Hz (by default), to exclude unwanted noise such as electricity grid frequency (60 Hz in Canada) or sudden patients` moves (<1 Hz)
3) identify and remove intervals of special procedures performed on patients during recordings, such as hyperventilation (deep breathing) and photic stimulation (flashing light). Physicians apply these tests to patients in order to detect brain abnormal activity for epilepsy diagnosis. Since these procedures burst abnormal activity, and weren't performed for all subjects, we exclude them from the analysis. Also the recordings contain intervals with no signal. It is the results of turned off equipment or disconnected electrode. So we also have to avoid these flat intervals with zero signal. Thus traget slices acquired only from clean intervals from each EEG, without flat intervals, hyperventilation and photic stimulation. Slices taken from the beginning, first minute taken as "bad" by default. The algoritm also handles cases when "bad intervals" overlap
4) **In case of extracting to Numpy arrays** signal values are also ZScore noramilized. Doesn't apply in case of saving output to EDF file(s).
## Usage
You need both modules edf_preprocessing.py and individual_func.py. The later contains python routine for saving output in EDF format again. The sample code for testing this out is given in Pipeline1.py. (Also shown here)

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from edf_extraction import Extractor

filepath = "C:/Srirupa/EEG Prepocessing/Raw EEG/sample_data.edf"
output_path = "C:/Srirupa/EEG Prepocessing/Clean EEG"

# Initiate the preprocessing(cleaning) object, filter the data between 0.5 Hz and 55 Hz and resample to 500 Hz.
p = Extractor(filepath, target_frequency=500)

# This calls internal functions to detect 'bad intervals' and define 5 'good' ones 60 seconds each
p.extract_good(target_length=60, target_segments=5)

# Calling the function saves new clean EDF files to output_folder. In case there are more than 1, it adds suffix "_n" to the file name 
p.save_edf(folder=output_path, filename='clean_data.edf')

```

## Pipeline 2 
This is the next step after Pipeline 1. Can only be implemented once Pipeline1 has been executed.

1) resample each recording's signal to traget frequency
2) filters the EEG segemts from a lower frequency of 5 Hz to an upper frequency of 100 Hz
3) marks the "bad" channels as per the PREP pipeline
4) performs ICA on the given EEG segment for both EOG and ECG channels
5) Then the object returns a pre-processed EEG data in raw format

## Usage
You need the module preprocessing.py. The sample code for testing this out is given in Pipeline2.py. (Also shown here)

By default veiwplots is False in case of multiple files. To view the plots you need to pass True as the third argument(usually done in case of using single file inputs).

```python
import os
import mne
from mne import viz
from individual_func import write_mne_edf
from preprocessing import Pipeline

filepath = "C:/Srirupa/EEG Prepocessing/Clean EEG/clean_data_1.edf"
output_path = "C:/Srirupa/EEG Prepocessing/Processed EEG/processed_data_1.edf"

# Initiate the preprocessing object
p = Pipeline(filepath, True)

# Calling the function filters the data between 0.5 Hz and 55 Hz, resamples to 500 Hz
# and performs ICA after applying the PREP pipeline to remove bad channels
p.applyPipeline(500, 12, True)

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

```

## Pipeline 3 for extraction and preprocessing from multiple files
Preprocessing, cleaning and saving new data to multiple EDF files into target folder

## Usage
You need modules multiple_preprocessing.py and edf_extraction.py, which are in turn dependant on modules preprocessing.py and individual_func.py. The sample code for testing this out is given in Pipeline3.py. (Also shown here)
```python
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

```