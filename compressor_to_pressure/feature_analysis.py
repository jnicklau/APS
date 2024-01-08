# feature_analysis.py
import matplotlib.pyplot as plt
import numpy as np

from reading_data import *
from filenames import *


def time_frequency_analysis(s, maxy=1e6, pbool=True):
    """
    returns and plots a fft of the signal s
    s: pd.Series
    """
    fft = np.fft.fft(s)
    f_per_dataset = np.arange(0, len(fft))
    n_samples_sec = len(s)
    seconds_per_day = 60 * 60 * 24
    seconds_per_dataset = n_samples_sec / seconds_per_day
    f_per_day = f_per_dataset / seconds_per_dataset
    # print(np.abs(fft).min(),np.abs(fft).max())
    if pbool == True:
        plt.step(f_per_day, np.abs(fft))
        plt.xscale("log")
        plt.ylim([0, np.abs(fft[1:-2]).max() * 1.1])
        # plt.ylim([0,1e6])
        plt.xlim([0.02, seconds_per_day])
        plt.xticks(
            [0.143, 1, 24, 60 * 24, seconds_per_day],
            labels=["1/week", "1/day", "1/hour", "1/minute", "1/second"],
        )
        plt.title("%s FFT" % s.name)
        plt.show()
    return fft


# ===========================================================
if __name__ == "__main__":
    # ------------------------------------------------------------
    print("feature_analysis")
