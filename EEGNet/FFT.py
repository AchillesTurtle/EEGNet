from scipy.signal import firwin, lfilter, filtfilt, butter,stft
from sklearn.preprocessing import scale,StandardScaler
import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
from scipy.signal import freqz

fs = 200
lowcut = 1
highcut = 40
sample_rate = fs
ntaps = 512
downsample_flag = True  # default is 200Hz
downsample_freq = 128  # Hz
trial_n = int(1.3 * downsample_freq)-1
"""
def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps
taps_hamming = bandpass_firwin(ntaps, lowcut, highcut, fs=fs)
delay = 0.5 * (ntaps - 1)
"""


def butter_bandpass(lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=8):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data,axis=0)
    return y
"""
plt.figure(1)
plt.clf()
for order in [3, 6, 8]:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = freqz(b, a, worN=2000)
    plt.plot(((fs * 0.5 / np.pi) * w)[0:1200], abs(h)[0:1200], label="order = %d" % order)

plt.plot([0, 0.3 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
         '--', label='sqrt(0.5)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')
plt.show()
input()
"""
def get_channel_index():
    #tot_channels = ['Fp1','Fp2','AF7','AF3','AF4','AF8','F7','F5','F3','F1','Fz','F2','F4',
    #                'F6','F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','C5',
    #               'C3','C1','Cz','C2','C4','C6','CPz']
    #tot_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
    #                'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP5', 'CP1', 'CP2', 'CP6', 'TP8',
    #                'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'POz', 'P08', 'O1', 'O2']
    channel_index = range(1,57)
   # with open(os.path.join("ner2015", "Data_S02_Sess01.csv"), 'r') as ff:
   #     row1 = next(csv.reader(ff))
   #     for index, chan_name in enumerate(row1):
   #         if chan_name in tot_channels:
   #             channel_index.append(index)
    return channel_index

chan_num=56
train_sample_num = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
test_sample_num = [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]
chan_index = []
chan_index = get_channel_index()
train_data = np.empty([5440, 1, chan_num, trial_n], dtype=float)
test_data = np.empty([3400, 1, chan_num, trial_n], dtype=float)
print("Extracting input training data...")
for itera, sample in enumerate(train_sample_num):
    for test_num in range(1, 6):
        # generate numpy array and delete first row
        file_name = 'Data_S%02d_Sess%02d.csv' % (sample, test_num)
        print(file_name)
        data = pd.read_csv(os.path.join("ner2015", "train", file_name))
        data = data.values
        EventStampList = []
        for index, EventStamp in enumerate(data[:, 58]):  # index in excel minus 2
            if EventStamp:
                EventStampList.append(int(index*downsample_freq/200))

        #filtered_data = butter_bandpass_filter(data[:, 0:58], lowcut, highcut, fs, order=8)
        #scaler_filter=StandardScaler(with_std=False).fit(filtered_data)
        #filtered_data=scaler_filter.transform(filtered_data)
        filtered_data = butter_bandpass_filter(data[:, 0:58], lowcut, highcut, fs, order=8)
        downsampled_data=np.empty([int(len(data[:,0])*128/200),58])
        for n in range(int(len(data[:,0])*128/200)):
            downsampled_data[n,:] = filtered_data[int(n * 200 / downsample_freq)]
        # extract and shape data as input
        train_index = itera * 340 + (test_num - 1) * 60
        for index, event_stamp in enumerate(EventStampList):
            segment = np.reshape(np.transpose(downsampled_data[event_stamp:event_stamp + trial_n, chan_index]), (1, 1, chan_num, trial_n))
            """
            plt.plot(data[event_stamp:event_stamp + 260,chan_index[0]], 'k-', label='org')
            plt.plot(filtered_data[event_stamp:event_stamp + 260,chan_index[0]], 'b-', label='filtered')
            plt.legend(loc='best')
            plt.show()
            input()
            """
            segment = np.fft.fft(segment)
            train_data[index + train_index, 0, :, :] = segment



print("Extracting input testing data...")
for itera, sample in enumerate(test_sample_num):
    for test_num in range(1, 6):
        # generate numpy array and delete first row
        file_name = 'Data_S%02d_Sess%02d.csv' % (sample, test_num)
        print(file_name)
        data = pd.read_csv(os.path.join("ner2015", "test", file_name))
        data = data.values
        EventStampList = []
        for index, EventStamp in enumerate(data[:, 58]):  # index in excel minus 2
            if EventStamp:
                EventStampList.append(int(index*downsample_freq/200))

        #filtered_data = butter_bandpass_filter(data[:, 0:58], lowcut, highcut, fs, order=8)
        #scaler_filter=StandardScaler(with_std=False).fit(filtered_data)
        #filtered_data=scaler_filter.transform(filtered_data)
        filtered_data = butter_bandpass_filter(data[:, 0:58], lowcut, highcut, fs, order=8)
        downsampled_data=np.empty([int(len(data[:,0])*128/200),58])
        for n in range(int(len(data[:,0])*128/200)):
            downsampled_data[n,:] = filtered_data[int(n * 200 / downsample_freq)]
        # extract and shape data as input
        test_index = itera * 340 + (test_num - 1) * 60
        for index, event_stamp in enumerate(EventStampList):
            segment = np.reshape(np.transpose(downsampled_data[event_stamp:event_stamp + trial_n, chan_index]), (1, 1, chan_num, trial_n))
            segment = np.fft.fft(segment)
            test_data[index + test_index, 0, :, :] = segment


train_data = train_data.reshape(5440 * chan_num, trial_n)
test_data = test_data.reshape(3400 * chan_num, trial_n)
np.savetxt("56_FFT_train_data_downsampled %d.csv" % downsample_freq, train_data, delimiter=",")
np.savetxt("56_FFT_test_data_downsampled %d.csv" % downsample_freq, test_data, delimiter=",")
print('Filtered successfully!')
