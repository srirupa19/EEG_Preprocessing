import numpy as np
import pandas as pd
import os
from individual_func import write_mne_edf
import mne
from mne.preprocessing import annotate_amplitude
from individual_func import write_mne_edf
from preprocessing import Pipeline

def get_processed_data(source_folder, target_folder, target_frequency, n_components, nfiles=None):
    """ The function run a pipeline for applying the preprocessing steps, namely, filrering,
    re-sampling, removing bad chaanels and performing ICA on multiple EDF files. It takes preprocessing 
    parameters, look up for the files in source folder, and perform preprocessing if found.
    
    Args:
        source_folder: folder path with clean extractd segments of EDF files 
        target_folder: folder where to save pre-proceesed in EDF formats
        target_frequency: interger indicating the final EEG frequency after resampling
        n_components: number of components using which we will perform the ICA
        nfiles: limit number of files to preprocess and extract segments (default=None, no limit)
    
    """
   
    existing_edf_names = os.listdir(source_folder)

    i = 0

    for file in existing_edf_names:

        path = source_folder + '/' + file

        try:

            # Initiate the preprocessing object
            p = Pipeline(path)

            # Calling the function filters the data between 0.5 Hz and 55 Hz, resamples to 500 Hz
            # and performs ICA after applying the PREP pipeline to remove bad channels
            p.applyPipeline(target_frequency, n_components)

            # Calling the function gets the pre-processed data in raw format
            raw = p.getRaw()

            # Calling the function drops the bad channels(as per PREP pipeline)
            raw.drop_channels(raw.info['bads'])

            # Calling the function saves pre-processed EDF files to output_folder.
            write_mne_edf(raw, fname=target_folder + '/processed_data_' + str(i + 1) + '.edf', overwrite=True)

            i += 1

        except:
            print('Preprocessing failed')

        if i % 100 == 0 and i != 0:
            print(i, 'EDF saved')

        if i == nfiles:
            break