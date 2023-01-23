import numpy as np
import pandas as pd
import os
from individual_func import write_mne_edf
import mne
from mne.preprocessing import annotate_amplitude


def read_edf(filepath):
    '''
    Read an EDF file with MNE package, creates the Raw EDF object. 
    Excludes some channels to keep only target ones.
    Prints warning in case the file doesn't have all 
    neeeded channels, doesn't return object in this case.
    
    Args:
      filepath: str with path to EDF file
    Returns:
      Raw EDF object
    '''
    # read EDF file while excluding some channels
    data = mne.io.read_raw_edf(filepath, exclude = ['A1', 'A2', 'AUX1', 
        'AUX2', 'AUX3', 'AUX4', 'AUX5', 'AUX6', 'AUX7', 'AUX8', 'Cz', 
        'DC1', 'DC2', 'DC3', 'DC4', 'DIF1', 'DIF2', 'DIF3', 'DIF4', 
        'Fp1', 'Fp2', 'Fpz', 'Fz', 'PG1', 'PG2', 'Patient Event', 'Photic', 
        'Pz', 'Trigger Event', 'X1', 'X2', 'aux1', 'phoic', 'photic'], verbose='warning', preload=True)
    
    if 'EKG1' in data.ch_names and 'EOG1' in data.ch_names:
        data.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog', 'EKG1': 'ecg', 'EKG2': 'ecg'})
        return data
    elif 'ECG1' in data.ch_names and 'EOG1' in data.ch_names:
        data.set_channel_types({'EOG1': 'eog', 'EOG2': 'eog', 'ECG1': 'ecg', 'ECG2': 'ecg'})
        return data
    else:
        print("Don't have needed channels")



class Extractor:
    """The class' aim is preprocessing clinical EEG recordings (in EDF format)
    and make them a suitable input for later analysis and ML applications.

    The class instantiates a preprocessing object which 
    carries a Raw EDF file through a sequence of operations: 
    (1) resample each recording's signal to traget frequency
    (2) identifies timestamps of hyperventilation (HV), photic stimulation (PhS)
    and flat (zero) signal (together - "bad" intervals)
    (3) extract EEG segment(s) of needed length from "good" intervals
    Then the object can save extracted segment(s) into new EDF files 
    OR return a Pandas DataFrame with data

    Attributes:
        filename: string with EDF file path
        target_frequency: interger indicating the final EEG frequency after resampling
        raw: MNE Raw EDF object
        sfreq: initial sampling frequency of EEG
        bad_intervals: list of lists of the form [start, end] with timemstamps in seconds,
            indicating starts and ends of bad interval (HV, PhS, flat signal)
        clean_intervals: list of lists of the form [start, end] with timemstamps in seconds,
            indicating starts and ends of segments to be extracted
            
    Methods:
        flat_intervals: returns list of [start, end] timestamps in seconds of zero signal
        hyperventilation: returns list of [start, end] timestamps in seconds of HV
        photic_stimulation: returns list of [start, end] timestamps in seconds of PhS
        extract_good: calling this method defines clean_intervals; 
            it doens't manipulate data itsef, just returns the intervals' timestamps
        save_edf: write new EDF files based on clean_intervals timestamps
    """
    
    def __init__(self, filepath, target_frequency):
        """
        Args:
            filepath: str with path to EDF file
            target_frequency: interger indicating the final EEG frequency after resampling
            lfreq: lower frequency boundary of the signal to keep
            hfreq: higher frequency boundary of the signal to keep
        """
        self.filename = filepath
        self.target_frequency = target_frequency
        self.raw = read_edf(filepath)
        self.sfreq = dict(self.raw.info)['sfreq']
        if(self.sfreq != self.target_frequency):
            self.raw.resample(self.target_frequency)
        
        self.clean_intervals = []
        self.intervals_df = pd.DataFrame()
        mne.set_log_level('warning')
        

    def flat_intervals(self):
        '''Identify beginning and end times of flat signal
        
        Returns:
            list of floats, contains start and end times
        '''
        annot_bad_seg, flat_chan = annotate_amplitude(self.raw, bad_percent=50.0, min_duration=10, flat=1e-06, picks=None, verbose=None)
        intervals = []
        for i in annot_bad_seg:
            start = list(i.items())[0][1]
            duration = list(i.items())[1][1]
            end = start+duration
            intervals.append([start,end])
        return intervals


    def hyperventilation(self):
        """Identify beginning and end of hyperventilation from EEG data

        Returns:
            list of floats, contains start and end times
        """

        start = np.nan
        end = np.nan

        for position, item in enumerate(self.raw.annotations.description):
            if item in ["HV 1Min", "HV 1 Min"]:
                start = self.raw.annotations.onset[position] - 90
            if item in ["Post HV 30 Sec", "Post HV 60 Sec", "Post HV 90 Sec"]:
                end = self.raw.annotations.onset[position] + (90 - int(item.split(' ')[2]))

        if np.isnan(start):
            for position, item in enumerate(self.raw.annotations.description):
                if item in ["HV Begin", "Begin HV"]:
                    start = self.raw.annotations.onset[position] - 30

        if np.isnan(end):
            for position, item in enumerate(self.raw.annotations.description):
                if item in ["HV End", "End HV"]:
                    end = self.raw.annotations.onset[position] + 90

        # when hyperventilation is present
        # eliminate the corresponding segment
        if start != np.nan and end != np.nan:
            return [[start, end]]
        else:
            return []

    def photic_stimulation(self):
        """Identify beginning and end times of photic stimulation.

        Returns:
            list of floats, contains start and end times
        """
        
        # store times when stimulation occurs
        stimulation = []
        
        # loop over descriptions and identify those that contain frequencies
        for position, annot in enumerate(self.raw.annotations.description):
            if "Hz" in annot:
                # record the positions of stimulations
                stimulation.append(position)
        
        # provided stimulation has occured
        if len(stimulation)>1:
            
            # identify beginning and end
            start = self.raw.annotations.onset[stimulation[0]]
            end = self.raw.annotations.onset[stimulation[-1]] + self.raw.annotations.duration[stimulation[-1]]
            return [[start, end]]    
        else:
            return []
        
        # null value when no stimulation is present
        return None


    def extract_good(self, target_length, target_segments):
        """ The function calls the functions above to identify "bad" intervals and
        updates the attribute clean_intervals with timesptamps to extract
        
        Args:
            target_length: length in seconds of the each 
                segments to extract from this EEG recording
            target_segments: number of segments to extract 
                from this EEG recording
                
        """
        
        self.bad_intervals = []
        # calling functions to identify different kinds of "bad" intervals
        self.bad_intervals.extend(self.flat_intervals())
        self.bad_intervals.extend(self.hyperventilation())
        self.bad_intervals.extend(self.photic_stimulation())
        self.bad_intervals.sort()
        
        self.clean_part = self.raw.copy()
        tmax = len(self.raw)/self.target_frequency
                
        # Add 'empty' bad intervals in the begging and in the end for furhter consistency
        self.bad_intervals.insert(0,[0, 420]) # <--- TAKE FIRST SEVEN MINUTES AS BAD BY DEFAULT
        self.bad_intervals.append([tmax, tmax])
        # Construct temporary dataframe to find clean interval in EDF
        tmp_df = pd.DataFrame(self.bad_intervals, columns=['start', 'end'])
        
        # Define end of the clean interval as a start of next bad interval
        tmp_df['next_start'] = tmp_df['start'].shift(periods=-1)
        tmp_df.iloc[-1,-1] = tmax # <= Assign end of edf file as the end of last clean interval
        
        # Handle cases when bad intervals overlaps
        prev_value = 0
        new_ends = []
        for value in tmp_df['end']:
            if prev_value > value :
                new_ends.append(prev_value)
            else:
                new_ends.append(value)
                prev_value = value
        tmp_df['cumulative_end'] = new_ends
        
        # Calculate lengths of clean intervals
        tmp_df['clean_periods'] = tmp_df['next_start'] - tmp_df['cumulative_end']
        
        # Check whether there is at least 1 clean interval with needed target length
        if tmp_df[tmp_df['clean_periods'] >= target_length].shape[0] == 0:
            self.resolution = False
            pass
        else:    
            # if there is at least one clean segment of needed length, it updates clean_intervals list
            self.resolution = True
            
            # check how many availabe segments of needed length the whole recording has
            total_available_segments = (tmp_df[tmp_df['clean_periods'] > 0]['clean_periods'] // target_length).sum()
            
            # if we need 5 segments, and the recording has more, it extracts 5; 
            # if the recording has less than 5, let's say only 3 segments, it extracts 3
            if target_segments < total_available_segments:
                n_samples = target_segments
            else:
                n_samples = total_available_segments
                
            starts = list(tmp_df[tmp_df['clean_periods'] > 0]['cumulative_end'])
            n_available_segments = list(tmp_df[tmp_df['clean_periods'] > 0]['clean_periods'] // target_length)
            
            # updates clean_intervals attribute with timestamps
            # starting from the first available intervals
            for i in range(len(n_available_segments)):
                current_start = int(starts[i])
                for s in range(int(n_available_segments[i])):
                    self.clean_intervals.append(
                    (
                        int(current_start), 
                        int(current_start + target_length)
                    )
                    )
                    #print(s, self.clean_intervals)
                    current_start += target_length
                    if len(self.clean_intervals) >= n_samples:
                        break
                if len(self.clean_intervals) >= n_samples:
                    break
            
    def save_edf(self, folder, filename):
        """ The function write out new EDF file(s) based on clean_intervals timestamps.
        It save each segment into separate EDF file, with suffixes "[scan_id]_1",
        "[scan_id]_2", etc. 
        
        Args:
            folder - where to save new EDF files
            filename - main name for output files (suffix will be added for more > 1 files)
        """
        
        # check if there are available clean segments
        if self.resolution:
            for n in range(len(self.clean_intervals)):
                interval_start = self.clean_intervals[n][0]
                interval_end = self.clean_intervals[n][1]
                
                tmp_raw_edf = self.clean_part.copy()
                
                tmp_raw_edf.crop(interval_start, interval_end, include_tmax=False)
                
                if n >= 0:
                    scan_id = filename.split('.')[0]
                    write_mne_edf(tmp_raw_edf, fname=folder+'/'+scan_id + '_' + str(n + 1)+'.edf', overwrite=True)
                # else:
                #     write_mne_edf(tmp_raw_edf, fname=folder+'/'+filename, overwrite=True)
        else:
            print('No clean intervals of needed length')
            
            
def slice_edfs(source_folder, target_folder, target_frequency, 
               target_length, target_segments=1, nfiles=None):
    """ The function run a pipeline for extracting clean segment(s) of needed length 
    from multiple EDF files. It takes preprocessing parameters, look up for the files 
    in source folder, and perform preprocessing and extraction, if found.
    
    Args:
        source_folder: folder path with EDF files 
        target_folder: folder where to save extractd segments in EDF formats
        target_frequency: interger indicating the final EEG frequency after resampling
        target_length: length of each of the extracted segments (in seconds)
        target_segments: number of segments to extract from each EDF file;
            will extract less if less available (default=0.5)
        nfiles: limit number of files to preprocess and extract segments (default=None, no limit)
    
    """
   
    existing_edf_names = os.listdir(source_folder)

    i = 0

    for file in existing_edf_names:

        path = source_folder + '/' + file

        try:
            # Initiate the preprocessing object, resample and filter the data
            e = Extractor(path, target_frequency=target_frequency)

            # This calls internal functions to detect 'bad intervals' and define 5 'good' ones 60 seconds each
            e.extract_good(target_length=target_length, target_segments=target_segments)

            # Calling the function saves new EDF files to output_folder. In case there are more than 1, it adds suffix "_n" to the file name 
            e.save_edf(folder=target_folder, filename=file)

            i += 1

        except:
            print('Extraction failed')

        if i % 100 == 0 and i != 0:
            print(i, 'EDF saved')

        if i == nfiles:
            break