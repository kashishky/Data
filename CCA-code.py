import numpy as np
from sklearn.cross_decomposition import CCA
import pandas as pd

# Code for CCAAnalysis taken from https://github.com/Mentalab-hub/explorepy/blob/master/examples/ssvep_demo/analysis.py
class CCAAnalysis:
    """Canonical Correlation Analysis for SSVEP paradigm"""
    def __init__(self, freqs, win_len, s_rate, n_harmonics=1):
        """
        Args:
            freqs (list): List of target frequencies
            win_len (float): Window length
            s_rate (int): Sampling rate of EEG signal
            n_harmonics (int): Number of harmonics to be considered
        """
        self.freqs = freqs
        self.win_len = win_len
        self.s_rate = s_rate
        self.n_harmonics = n_harmonics
        self.train_data = self._init_train_data()
        self.cca = CCA(n_components=1)

    def _init_train_data(self):
        t_vec = np.linspace(0, self.win_len, int(self.s_rate * self.win_len))
        targets = {}
        for freq in self.freqs:
            sig_sin, sig_cos = [], []
            for harmonics in range(self.n_harmonics):
                sig_sin.append(np.sin(2 * np.pi * harmonics * freq * t_vec))
                sig_cos.append(np.cos(2 * np.pi * harmonics * freq * t_vec))
            targets[freq] = np.array(sig_sin + sig_cos).T
        return targets

    def apply_cca(self, eeg):
        """Apply CCA analysis to EEG data and return scores for each target frequency

        Args:
            eeg (np.array): EEG array [n_samples, n_chan]

        Returns:
            list of scores for target frequencies
        """
        scores = []
        for key in self.train_data:
            sig_c, t_c = self.cca.fit_transform(eeg, self.train_data[key])
            scores.append(np.corrcoef(sig_c.T, t_c.T)[0, 1])
        return scores

# Main method edited, with specific data
if __name__ == '__main__':
    freqs = [13.333, 15, 17.142, 14.118, 16] # twice the stimulation frequencies COM 1,2,3,4,5
    t_len = 2 # 2 second window
    s_rate = 125
    #t_vec = np.linspace(0, t_len, s_rate * t_len)

    class_names = ['BottomLeft', 'BottomMiddle', 'BottomRight', 'TopLeft', 'TopRight']

    # Reading one trial from each stimulus for GG subject
    fnames = ['/Users/siddhant/Documents/Neurotech-X-Drone/Data/SSVEP/GG/' + i + '/64sec_1.csv' for i in class_names]
    dfs = [pd.read_csv(name) for name in fnames]

    for i,df in enumerate(dfs):
        # test_sig = np.sin(2 * np.pi * 10 * t_vec) + 0.05 * np.random.rand(len(t_vec))
        # Looking at channel 15, with a 2 second window 251:501 which is 250 datapoints
        test_sig = np.array(list(df.loc[14,:])[251:501]) 
        cca_analysis = CCAAnalysis(freqs=freqs, win_len=t_len, s_rate=s_rate, n_harmonics=2)
        r = cca_analysis.apply_cca(np.array(test_sig)[:, np.newaxis])

        # Printing the max correlation coefficient across all stimulus frequencies, the Command Number (1-5), the Predicted Stimulus Frequency (highest correlated) from CCA, The Actual Stimulus that was tested 
        print(np.max(r), np.argmax(r)+1, 'Pred', freqs[np.argmax(r)], 'Actual', class_names[i])
