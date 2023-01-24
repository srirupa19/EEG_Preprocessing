import mne
import numpy as np
import os
import matplotlib.pyplot as plt
from mne import viz
from pyprep.prep_pipeline import PrepPipeline
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
class Pipeline:
    """The class' aim is preprocessing clean extracted segments of clinical 
    EEG recordings (in EDF format) and make them a suitable input for later analysis 
    and ML applications.

    The class instantiates a preprocessing object which 
    carries a Raw EDF file through a sequence of operations: 
    (1) resample each recording's signal to traget frequency
    (2) filters the EEG segemts from a lower frequency of 5 Hz to an upper frequency of 100 Hz
    (3) marks the "bad" channels as per the PREP pipeline
    (4) performs ICA on the given EEG segment for both EOG and ECG channels
    Then the object returns a pre-processed EEG data in raw format

    Attributes:
        rawResampled: contains the resampled EEG segment in raw format
        rawFiltered: contains the filtered EEG segment in raw format
        raw: MNE Raw EDF object
        rawPrep: contains the EEG segment in raw format after marking bad channels(as per PREP)
        rawIca: contains the EEG segment after ICA is performed on it for both EOG and ECG channels
            
    Methods:
        resample: resamples the EEG segment to the target_frequency
        filter: filters the EEG signal from 5 Hz to 100 Hz
        prep: applies the PREP pipeline the mark the bad channels
        ica: performs ICA on the given EEG segment for both EOG and ECG channels
        showplot: shows the time domain plot of the given EEG segment
        applyPipeline: applies the pipeline (resampling, filtering, applying PREP, performing ICA)
                        on the given EEG segment
        getRaw: returns the preprocessed EEG segment in raw format

    """

    def __init__(self, file_name, view_plots = False) -> None:
        """
        Args:
            file_name: str with path to EDF file
            view_plots: boolean value to denote if we want to view plots. 
                        Turned off while processing multiple files. 
        """
        self.raw = mne.io.read_raw_edf(file_name)
        if 'EKG1' in self.raw.ch_names and 'EOG1' in self.raw.ch_names:
            self.raw.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog', 'EKG1': 'ecg', 'EKG2': 'ecg'})
        elif 'ECG1' in self.raw.ch_names and 'EOG1' in self.raw.ch_names:
            self.raw.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog', 'ECG1': 'ecg', 'ECG2': 'ecg'})
        
        self.rawResampled = None
        self.rawFiltered = None
        self.rawPrep = None
        self.rawIca = None

        print("Created raw object")
        if (view_plots):
            self.showplot(self.raw)
        
        
    def resample(self, target_frequency, view_plots) -> None:
        """Resamples the given EEG segement to the target frequency

        Args:
            target_frequency: interger indicating the final EEG frequency after resampling
            view_plots: boolean value to denote if we want to view plots

        Returns:
            None
        """
        self.rawResampled = self.raw
        self.rawResampled.resample(target_frequency)

        print("Resampled raw object")
        if (view_plots):
            self.showplot(self.rawResampled)


    def filter(self, view_plots) -> None:
        """Filters the given EEG segement between 5 Hz and 100 Hz

        Args:
            view_plots: boolean value to denote if we want to view plots

        Returns:
            None
        """
        self.rawFiltered = self.rawResampled
        channels = list(set(self.rawFiltered.ch_names) - set(["EOG1", "EOG2"]))

        self.rawFiltered.filter(l_freq=1, h_freq=100, picks=channels)
        self.rawFiltered.filter(l_freq=1, h_freq=5, picks=["EOG1", "EOG2"])

        print("Filtered raw object")
        if (view_plots):
            self.showplot(self.rawFiltered)

    def prep(self, view_plots) -> None:
        """Applies the PREP pipeline to the EEG segment to mark the bad channels

        Args:
            view_plots: boolean value to denote if we want to view plots

        Returns:
            None
        """
        self.rawPrep = self.rawFiltered
        mne.datasets.eegbci.standardize(self.rawPrep)

        # Add a montage to the data
        montage_kind = "standard_1005"
        montage = mne.channels.make_standard_montage(montage_kind)
        self.rawPrep.set_montage(montage, on_missing='ignore')

        # Extract some info
        sample_rate = self.rawPrep.info["sfreq"]

        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": np.arange(60, sample_rate / 2, 60),
        }

        prep = PrepPipeline(self.rawPrep, prep_params, montage)
        prep.fit()

        print("Bad channels: {}".format(prep.interpolated_channels))
        print("Bad channels original: {}".format(prep.noisy_channels_original["bad_all"]))
        print("Bad channels after interpolation: {}".format(prep.still_noisy_channels))
        print("SUCCESS Step 4: Applied Prep Pipeline to remove bad channels")

        self.rawPrep = prep.raw.copy()
        print("Applied ICA on raw object")

        if (view_plots):
            self.showplot(self.rawPrep)


    def ica(self, components, view_plots) -> None:
        """Performs ICA on the given EEG segment for both EOG and ECG channels

        Args:
            components: integer denoting the number of components to be used for performing ICA,
                        is usually taken same as the number of channels
            view_plots: boolean value to denote if we want to view plots

        Returns:
            None
        """
        self.rawIca = self.rawPrep
        self.rawIca.filter(l_freq=1, h_freq=None)
        raw_copy = self.rawIca.copy()
        ica = ICA(n_components=len(raw_copy.pick_types(eeg=True, exclude='bads', 
                selection=None, verbose=None).ch_names), max_iter='auto', random_state=97)
        ica.fit(self.rawIca)
        
        if "EOG1" in self.rawIca.ch_names or "EOG2" in self.rawIca.ch_names:
            eog_indices, eog_scores = ica.find_bads_eog(self.rawIca)
            ica.exclude = eog_indices

            if (view_plots) and eog_indices != []:
                ica.plot_scores(eog_scores)
            
                # plot diagnostics
                ica.plot_properties(self.rawIca, picks=eog_indices)

                # plot ICs applied to raw data, with EOG matches highlighted
                ica.plot_sources(self.rawIca, show_scrollbars=False)

                print("Removed EOG Artifacts using ICA")
                self.showplot(self.rawIca)

        if "EKG1" in self.rawIca.ch_names or "EKG2" in self.rawIca.ch_names:
            ica.exclude = []
            # find which ICs match the ECG pattern
            ecg_indices, ecg_scores = ica.find_bads_ecg(self.rawIca, method='correlation',
                                                        threshold='auto')
            ica.exclude = ecg_indices

            if (view_plots) and ecg_indices != []:
                # barplot of ICA component "ECG match" scores
                ica.plot_scores(ecg_scores)

                # plot ICs applied to raw data, with ECG matches highlighted
                ica.plot_sources(self.rawIca, show_scrollbars=False)
                print("Removed ECG Artifacts using ICA")

                # ica.apply(self.rawIca)                
                self.showplot(self.rawIca)


    def showplot(self, raw) -> None:
        """Shows the time domain plot of the given EEG segment for 30 seconds

        Args:
            raw: the EEG segment to be plotted 

        Returns:
            None
        """

        raw_np = raw.get_data();
        print(raw_np.shape);
        artifact_picks = mne.pick_channels(raw.ch_names, include=raw.ch_names)
        raw.plot(order=artifact_picks, n_channels=len(artifact_picks),
                show_scrollbars=False, duration=0.5, start=0, block=True)
        

    def applyPipeline(self, target_frequency, components, view_plots = False) -> None:
        """Applies the pipeline (resampling, filtering, applying PREP, performing ICA)
            on the given EEG segment

        Args:
            target_frequency: interger indicating the final EEG frequency after resampling
            components: integer denoting the number of components to be used for performing ICA,
                        is usually taken same as the number of channels
            view_plots: boolean value to denote if we want to view plots 

        Returns:
            None
        """
        self.resample(target_frequency, view_plots)
        self.filter(view_plots)
        self.prep(view_plots)
        self.ica(components, view_plots) 

    def getRaw(self):
        """
        Returns:
            Raw EDF object (preprocessed)
        """
        return self.rawIca
        
