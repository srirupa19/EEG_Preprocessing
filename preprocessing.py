import mne
import numpy as np
import os
import matplotlib.pyplot as plt
from mne import viz
from pyprep.prep_pipeline import PrepPipeline
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
class Pipeline:
    def __init__(self, file_name) -> None:
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
        self.showplot(self.raw)
        
    def resample(self, target_frequency):
        self.rawResampled = self.raw
        self.rawResampled.resample(target_frequency)

        print("Resampled raw object")
        self.showplot(self.rawResampled)


    def filter(self):
        self.rawFiltered = self.rawResampled
        channels = list(set(self.rawFiltered.ch_names) - set(["EOG1", "EOG2"]))

        self.rawFiltered.filter(l_freq=1, h_freq=100, picks=channels)
        self.rawFiltered.filter(l_freq=1, h_freq=5, picks=["EOG1", "EOG2"])

        print("Filtered raw object")
        self.showplot(self.rawFiltered)

    def prep(self):
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
        self.showplot(self.rawPrep)


    def ica(self, components):
        self.rawIca = self.rawPrep
        self.rawIca.filter(l_freq=1, h_freq=None)
        ica = ICA(n_components=components, max_iter='auto', random_state=97)
        ica.fit(self.rawIca)
        
        if "EOG1" in self.rawIca.ch_names or "EOG2" in self.rawIca.ch_names:
            eog_indices, eog_scores = ica.find_bads_eog(self.rawIca)
            ica.exclude = eog_indices

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

            # barplot of ICA component "ECG match" scores
            ica.plot_scores(ecg_scores)

            # plot ICs applied to raw data, with ECG matches highlighted
            ica.plot_sources(self.rawIca, show_scrollbars=False)
            print("Removed ECG Artifacts using ICA")

            # ica.apply(self.rawIca)
            self.showplot(self.rawIca)


    def showplot(self, raw):
        artifact_picks = mne.pick_channels(raw.ch_names, include=raw.ch_names)
        raw.plot(order=artifact_picks, n_channels=len(artifact_picks),
                show_scrollbars=False, duration=0.5, start=0, block=True)
        

    def applyPipeline(self, target_frequency, components):
        self.resample(target_frequency)
        self.filter()
        self.prep()
        self.ica(components) 

    def getRaw(self):
        return self.rawPrep
        
