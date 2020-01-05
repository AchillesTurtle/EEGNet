##################################################
####Trains EEGNet with 16 data sets, tests on 10 datasets###
##################################################

import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from keras import backend as K
from keras.models import load_model
from keras.utils import np_utils

K.set_image_dim_ordering('th')  # set dimension ordering to theano

# import models
from EEGModels import EEGNet, EEGNet2
from EEGModels import DeepConvNet

# individuals number
train_sample_num = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
test_sample_num = [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]
train_subject_num = len(train_sample_num)
test_subject_num = len(test_sample_num)
train_tot_trials = int(train_subject_num * 340)
test_tot_trials = int(test_subject_num * 340)
# variables
batchsize = 34
sample_rate = 128  # Hz
trial_time = 1.1  # s
trial_n = int(trial_time * sample_rate)
channels_n = 56  # depends on preproccessed file
shift = 8  # samples


# return wanted channels in number index
def get_channel_index():
    tot_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
                    'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP5', 'CP1', 'CP2', 'CP6', 'TP8',
                    'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'POz', 'P08', 'O1', 'O2']
    channel_index = []
    with open(os.path.join("ner2015", "Data_S02_Sess01.csv"), 'r') as ff:
        row1 = next(csv.reader(ff))
        for index, chan_name in enumerate(row1):
            if chan_name in tot_channels:
                channel_index.append(index)
    return channel_index


#####################
#######LABELS######
#####################
print("Extracting data labels...")
data = np.genfromtxt(os.path.join("ner2015", "TrainLabels.csv"), delimiter=',')  # needs to be replace by pandas
train_labels = data[1:train_tot_trials + 1, 1]  # omiting first row and column
train_target = []
train_nontarget = []
# balancing training target/nontarget labels
# where we assume target trials > nontarget trials
for subject in range(train_subject_num):
    for itera in range(340):
        if train_labels[subject * 340 + itera]:
            train_target.append(subject * 340 + itera)
        else:
            train_nontarget.append(subject * 340 + itera)
random.shuffle(train_target)
train_target = train_target[0:len(train_nontarget)]
train_balanced_list = (train_target + train_nontarget)
train_balanced_list.sort()
# getting test labels
data = np.genfromtxt(os.path.join("ner2015", "TestLabels.csv"), delimiter=',')
test_labels = data

# categorical - turning into network output format
flatten_train_labels = train_labels[train_balanced_list]
flatten_test_labels = test_labels
train_labels = np_utils.to_categorical(train_labels, 2)
test_labels = np_utils.to_categorical(test_labels, 2)
train_labels = train_labels[train_balanced_list, :]
"""
new_flatten_train_labels=np.empty([len(flatten_train_labels)*2])
for n in range(len(flatten_train_labels)):
    new_flatten_train_labels[n*2]=flatten_train_labels[n]
    new_flatten_train_labels[n*2+1]=flatten_train_labels[n]
new_train_labels=np_utils.to_categorical(new_flatten_train_labels, 2)
"""

########################
########EEGDATA#######
########################
# Was preprocessed by dataextraction.py(modified) or bandpass.py(filtered)
train_data = np.empty([len(train_balanced_list), 1, channels_n, trial_n], dtype=float)
test_data = np.empty([test_tot_trials, 1, channels_n, trial_n], dtype=float)
print("Extracting train data...")
data = np.genfromtxt("56_nn_train_data_filtered_downsampled 128.csv", delimiter=',')
for index, trial in enumerate(train_balanced_list):
    train_data[index, 0, :, :] = data[trial * channels_n:(trial + 1) * channels_n, :trial_n]
    # train_data[index*2, 0, :, :] = data[trial * channels_n:(trial + 1) * channels_n, :trial_n]
    # train_data[index*2+1, 0, :, :] = data[trial * channels_n:(trial + 1) * channels_n, shift:trial_n+shift]

print("Extracting test data...")
data = np.genfromtxt("56_nn_test_data_filtered_downsampled 128.csv", delimiter=',')
for index in range(test_tot_trials):
    test_data[index, 0, :, :] = data[index * channels_n:(index + 1) * channels_n, :trial_n]

#############################
#########MODEL############
############################
# for kernel_trial in kernel_list:
#    for dropout_trial in dropout_list:
if (False):  # load previous model
    model = load_model(' ')
# model=load_model(input('Model name:')+'.h5')
else:
    # Define and Train model
    # model = DeepConvNet((nb_classes, Chans = 32, Samples = trial_n,dropoutRate = 0.5):
    model = EEGNet(nb_classes=2, Chans=56, Samples=trial_n, regRate=0.001, dropoutRate=0.25, kernels=[(2, 32), (8, 4)])
    # model = EEGNet(nb_classes=2, Chans=32, Samples=trial_n, dropoutRate=0.4, kernels=[(2, 64), (8, 32)])
    # model = EEGNet(nb_classes=2, Chans=32, Samples=260, dropoutRate=dropout_trial, kernels=kernel_trial)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_data, train_labels, batch_size=batchsize, epochs=500, verbose=2)
    auc_values = []
    """
    for n in range(0,500,10):
        model.fit(train_data, train_labels, batch_size=batchsize, epochs=10, verbose=2)
        test_predict = model.predict(test_data, batch_size=batchsize, verbose=0)
        fpr, tpr, thresholds = roc_curve(flatten_test_labels, test_predict[:, 1], pos_label=1)
        test_auc_value = auc(fpr, tpr)
        auc_values.append(test_auc_value)
plt.plot(range(0,500,10),auc_values)
plt.title('Test AUC for filtered EEG')
plt.ylabel('AUC')
plt.xlabel('epoch')
plt.show()
"""
# model.save('***.h5')
# generate prediction probabilities
train_predict = model.predict(train_data, batch_size=batchsize, verbose=0)
test_predict = model.predict(test_data, batch_size=batchsize, verbose=0)
print(model.evaluate(test_data, test_labels, batch_size=batchsize))

# Calculate AUC
fpr, tpr, thresholds = roc_curve(flatten_train_labels, train_predict[:, 1], pos_label=1)
print("Training auc:")
train_auc_value = auc(fpr, tpr)
print(train_auc_value)

fpr, tpr, thresholds = roc_curve(flatten_test_labels, test_predict[:, 1], pos_label=1)
print("Testing auc:")
test_auc_value = auc(fpr, tpr)
print(test_auc_value)

del model

# plot AUC curve of test
lw = 2
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='Testing ROC curve (area = %0.2f)' % test_auc_value)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
