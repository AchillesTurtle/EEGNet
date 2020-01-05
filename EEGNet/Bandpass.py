####################################################
##trial extraction, filtering, and downsampling for dataset in
##Perrin, M., Maby, E., Daligault, S., Bertrand, O., & Mattout, J.
##Objective and subjective evaluation of online error correction during P300-based spelling. Advances in Human-Computer Interaction, 2012, 4.
###################################################
from scipy.signal import firwin, lfilter, filtfilt, butter, stft
from sklearn.preprocessing import scale, StandardScaler
import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
from scipy.signal import freqz

fs = 200  # default sampling rate of data
lowcut = 1
highcut = 40
sample_rate = fs
ntaps = 512
downsample_flag = True  # to down_sample or not
downsample_freq = 128  # Hz
trial_n = int(1.3 * downsample_freq)  # length(samples) of single trial with downsampled freq (1.3s)
chan_num = 56 #number of channels

# butter worth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y


#gets index number for certain channels
def get_channel_index():
    # tot_channels = ['Fp1','Fp2','AF7','AF3','AF4','AF8','F7','F5','F3','F1','Fz','F2','F4',
    #                'F6','F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6','FT8','C5',
    #               'C3','C1','Cz','C2','C4','C6','CPz']
    #tot_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
    #                'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP5', 'CP1', 'CP2', 'CP6', 'TP8',
    #                'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'POz', 'P08', 'O1', 'O2']
    channel_index = range(1,57)
    #channel_index = []
    #with open(os.path.join("ner2015", "train","Data_S02_Sess01.csv"), 'r') as ff:
    #    row1 = next(csv.reader(ff))
    #    for index, chan_name in enumerate(row1):
    #        if chan_name in tot_channels:
    #            channel_index.append(index)
    return channel_index


train_sample_num = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
test_sample_num = [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]
chan_index = []
chan_index = get_channel_index()
train_data = np.empty([len(train_sample_num)*340, 1, chan_num, 260], dtype=float)
test_data = np.empty([len(test_sample_num)*340, 1, chan_num, 260], dtype=float)
print("Extracting input training data...")
for itera, sample in enumerate(train_sample_num):
    for test_num in range(1, 6):
        # generate numpy array and delete first row
        file_name = 'Data_S%02d_Sess%02d.csv' % (sample, test_num)
        print(file_name)
        data = pd.read_csv(os.path.join("ner2015", "train", file_name))
        data = data.values
        EventStampList = []
        for index, EventStamp in enumerate(data[:, 58]):  # get event stamp
            if EventStamp:
                EventStampList.append((int)(index))

        # filtered_data = butter_bandpass_filter(data[:, 0:58], lowcut, highcut, fs, order=8)
        # scaler_filter=StandardScaler(with_std=False).fit(filtered_data)
        # filtered_data=scaler_filter.transform(filtered_data)
        filtered_data = butter_bandpass_filter(data[:, 0:58], lowcut, highcut, fs, order=4)
        # extract and shape data as input
        train_index = itera * 340 + (test_num - 1) * 60
        for index, event_stamp in enumerate(EventStampList):
            segment = np.reshape(np.transpose(filtered_data[event_stamp:event_stamp + 260, chan_index]),
                                 (1, 1, chan_num, 260))
            train_data[index + train_index, 0, :, :] = segment
            """
            plt.plot(data[event_stamp:event_stamp + 260,chan_index[0]], 'k-', label='org')
            plt.plot(filtered_data[event_stamp:event_stamp + 260,chan_index[0]], 'b-', label='filtered')
            plt.legend(loc='best')
            plt.show()
            input()
"""


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
                EventStampList.append((int)(index))

        # filtered_data = butter_bandpass_filter(data[:, 0:58], lowcut, highcut, fs, order=8)
        # scaler_filter=StandardScaler(with_std=False).fit(filtered_data)
        # filtered_data=scaler_filter.transform(filtered_data)
        filtered_data = butter_bandpass_filter(data[:, 0:58], lowcut, highcut, fs, order=8)
        # extract and shape data as input
        test_index = itera * 340 + (test_num - 1) * 60
        for index, event_stamp in enumerate(EventStampList):
            segment = np.reshape(np.transpose(filtered_data[event_stamp:event_stamp + 260, chan_index]),
                                 (1, 1, chan_num, 260))
            test_data[index + test_index, 0, :, :] = segment

if downsample_flag:
    print('Start Downsampling to %d Hz' % downsample_freq)
    downsampled_train_data = np.empty([len(train_sample_num)*340, 1, chan_num, trial_n], dtype=float)
    for index in range(len(train_sample_num)*340):
        for n in range(trial_n):
            downsampled_train_data[index, 0, :, n] = train_data[index, 0, :, int(n * 200 / downsample_freq)]
    downsampled_test_data = np.empty([len(test_sample_num)*340, 1, chan_num, trial_n], dtype=float)
    for index in range(len(test_sample_num)*340):
        for n in range(trial_n):
            downsampled_test_data[index, 0, :, n] = test_data[index, 0, :, int(n * 200 / downsample_freq)]
    downsampled_train_data = downsampled_train_data.reshape(len(train_sample_num)*340 * chan_num, trial_n)
    downsampled_test_data = downsampled_test_data.reshape(len(test_sample_num)*340 * chan_num, trial_n)
    np.savetxt("56_nn_train_data_filtered_downsampled %d.csv" % downsample_freq, downsampled_train_data, delimiter=",")
    np.savetxt("56_nn_test_data_filtered_downsampled %d.csv" % downsample_freq, downsampled_test_data, delimiter=",")
    print('Down sampled to %d' % downsample_freq)
else: #without downsampling
    train_data = train_data.reshape(len(train_sample_num)*340 * chan_num, 260)
    test_data = test_data.reshape(len(test_sample_num)*340 * chan_num, 260)
    np.savetxt("56_nn_train_data_filtered.csv", train_data, delimiter=",")
    np.savetxt("56_nn_test_data_filtered.csv", test_data, delimiter=",")
print('Filtered successfully!')
