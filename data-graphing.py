from re import S
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import pandas as pd
import os
from scipy.signal import butter, lfilter, find_peaks


path = "/Users/kashi/Documents/GitHub/Neurotech-X-Drone/Data/SSVEP/"
# You can add more subjects to the list
subjects = ['GG/']#, 'JP/', 'MC/'] # GB doesn't have anything, NG only has TopLeft
stimuli = ['BottomLeft/', 'BottomMiddle/', 'BottomRight/','TopLeft/','TopRight/']

all_paths = [path + sub + stim for sub in subjects for stim in stimuli]

list_of_csv_paths = []
for p in all_paths:
    # Trial 1 only selected. Use csv_files = os.listdir(p) if you want all trials
    csv_files =  ['64sec_1.csv'] 
    csv_paths = [p + c for c in csv_files]
    list_of_csv_paths.append(csv_paths)

all_files = list(np.concatenate(list_of_csv_paths).flat)

# Butterworth bandpass filter - causing ringing in filtered signal 
def butter_bandpass(low, high, fs, order = 5):
    nyq = 0.5 * fs
    low_f = low/nyq
    high_f = high/nyq
    b, a = butter(order, [low_f,high_f], btype='band')
    return b,a 

def butter_bp_filter(data, low, high, fs, order = 5):
    b, a = butter_bandpass(low, high, fs, order = order)
    y = lfilter(b,a,data)
    return y

for fp in all_files:
    df = pd.read_csv(fp)
    dt = 0.008 # 1/125
    time_window = 3 # Can try different windows to visualise
    t = np.arange(0, time_window, dt) 
    #n_channels = 16
    # manually selecting important channels near occipital lobe
    imp_channels = [4,5,6,7,14,15] # 0-indexed: Ch 5,6,7,8,15,16
    tok = fp.split('/')

    for i in imp_channels:
        s = list(df.loc[i,:])[1:376]
        filtered_s = butter_bp_filter(s, 5,55,125,order=4) # filtering with Bandpass
        
        # original signal
        plt.subplot(221)
        plt.plot(t, s)
        # psd of original signal
        plt.subplot(222)
        plt.psd(s, NFFT = 375, Fs = 1 / dt, label="Channel " + str(i+1))

        # filtered signal
        plt.subplot(223)
        plt.plot(t, filtered_s)
        # psd of filtered signal
        plt.subplot(224)
        plt.psd(filtered_s, NFFT = 375, Fs = 1 / dt, label="Channel " + str(i+1))
        
        # Use plt.show() here if you want to see separate plots per channel, and comment out the code after this line
        

        
    # This is the plot for all 6 channels, comment out and use plt.show() in loop as mentioned above for per-channel graphs
    title = tok[-3] + ' ' + tok[-2] + ' ' + tok[-1]
    plt.title(title)
    plt.legend()
    plt.show()
