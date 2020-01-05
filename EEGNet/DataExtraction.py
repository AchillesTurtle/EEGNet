import os
import csv
import numpy as np
import pandas as pd
train_sample_num=[2,6,7,11,12,13,14,16,17,18,20,21,22,23,24,26]
test_sample_num=[1,3,4,5,8,9,10,15,19,25]
#return channels in number index
def get_channel_index():
    tot_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
                    'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP5', 'CP1', 'CP2', 'CP6', 'TP8',
                    'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'POz', 'P08', 'O1', 'O2']
    channel_index=[]
    with open(os.path.join("ner2015", "Data_S02_Sess01.csv"), 'r') as ff:
        row1=next(csv.reader(ff))
        for index, chan_name in enumerate(row1):
            if chan_name in tot_channels:
                channel_index.append(index)
    return channel_index

chan_index=[]
chan_index=get_channel_index()
train_data = np.empty([5440, 1, 32, 260], dtype=float)
test_data = np.empty([3400,1,32,260],dtype=float)
print("Extracting input training data...")
for itera, sample in enumerate(train_sample_num):
    for test_num in range(1,6):
        #generate numpy array and delete first row
        file_name='Data_S%02d_Sess%02d.csv' % (sample,test_num)
        print(file_name)
        data=pd.read_csv(os.path.join("ner2015","train",file_name))
        data=data.values
        EventStampList=[]
        for index, EventStamp in enumerate(data[:,58]):  #index in excel minus 2
            if EventStamp:
                EventStampList.append(index)

        # extract and shape data as input
        train_index = itera * 340 + (test_num - 1) * 60
        for index, event_stamp in enumerate(EventStampList):
            segment=np.reshape(np.transpose(data[event_stamp:event_stamp+260,chan_index]),(1,1,32,260))
            train_data[index+train_index,0,:,:]=segment

print("Extracting input testing data...")
for itera, sample in enumerate(test_sample_num):
    for test_num in range(1,6):
        #generate numpy array and delete first row
        file_name='Data_S%02d_Sess%02d.csv' % (sample,test_num)
        print(file_name)
        data=pd.read_csv(os.path.join("ner2015","test",file_name))
        data=data.values
        EventStampList=[]
        for index, EventStamp in enumerate(data[:,58]):  #index in excel minus 2
            if EventStamp:
                EventStampList.append(index)

        # extract and shape data as input
        test_index = itera * 340 + (test_num - 1) * 60
        for index, event_stamp in enumerate(EventStampList):
            segment=np.reshape(np.transpose(data[event_stamp:event_stamp+260,chan_index]),(1,1,32,260))
            test_data[index+test_index,0,:,:]=segment



train_data=train_data.reshape(5440*32,260)
test_data=test_data.reshape(3400*32,260)
np.savetxt("train_data_modified.csv",train_data,delimiter=",")
np.savetxt("test_data_modified.csv",test_data,delimiter=",")