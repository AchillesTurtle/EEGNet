#########################################################################
####THIS CODE IS INTENDED TO REPRODUCE THE RESULTS OF EEGNET###
####Training with 12 data sets, validation with 4 datasets, and test with 10 datasets######
########################################################################
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
from EEGModels import EEGNet,EEGNetOrg
from EEGModels import DeepConvNet

# individuals number
train_sample_num = [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26]
test_sample_num = [1, 3, 4, 5, 8, 9, 10, 15, 19, 25]
train_subject_num = len(train_sample_num)
test_subject_num = len(test_sample_num)
train_tot_trials = int(8 * 340)
test_tot_trials = int(test_subject_num * 340)
validation_tot_trials = int(4 * 340)
# variables
batchsize = 34
sample_rate = 128  # Hz
trial_time = 1.1  # s
trial_n = int(trial_time * sample_rate)
channels_n = 56  # depends on preproccessed file
shift=8



#####################
#######LABELS######
#####################
print("Extracting data labels...")
data = np.genfromtxt(os.path.join("ner2015", "TrainLabels.csv"), delimiter=',')  # needs to be replace by pandas
all_labels = data[1:, 1]  # omiting first row and column

# balancing training target/nontarget labels
# where we assume target trials > nontarget trials
# getting test labels
data = np.genfromtxt(os.path.join("ner2015", "TestLabels.csv"), delimiter=',')
test_labels = data
flatten_test_labels = test_labels
test_labels = np_utils.to_categorical(test_labels, 2)

# flatten_train_labels = np.argmax(train_labels, axis=1)  # flatten
# flatten_test_labels = np.argmax(test_labels, axis=1)


########################
########EEGDATA#######
########################
# Was preprocessed by dataextraction.py(modified) or bandpass.py(filtered)
#using pandas will be faster, genfromtxt is extremely slow
print("Extracting all data...") #train and validation
all_data = np.genfromtxt("56_nn_train_data_filtered_downsampled 128.csv", delimiter=',')

test_data = np.empty([test_tot_trials, 1, channels_n, trial_n], dtype=float)
print("Extracting test data...")
data = np.genfromtxt("56_nn_test_data_filtered_downsampled 128.csv", delimiter=',')
for index in range(test_tot_trials):    #formatting to right format
    test_data[index, 0, :, :] = data[index * channels_n:(index + 1) * channels_n, :trial_n]

#############################
#########MODEL############
############################
for fold in range(5):
    #choose subjects for training and validation randomly 8/4 (train/val)
    validation_list = random.sample(range(12), 4)
    validation_list.sort()
    training_list = []
    for ele in range(12):
        if ele not in validation_list:
            training_list.append(ele)

    train_target = []
    train_nontarget = []
    for subject in training_list:
        for itera in range(340):
            if all_labels[subject * 340 + itera]:
                train_target.append(subject * 340 + itera)
            else:
                train_nontarget.append(subject * 340 + itera)
    random.shuffle(train_target)
    train_target = train_target[0:len(train_nontarget)]
    train_balanced_list = (train_target + train_nontarget)
    train_balanced_list.sort()

    validation_labels = np.empty([4*340])
    for index,subject in enumerate(validation_list):
        validation_labels[index*340:(index+1)*340]=all_labels[subject * 340:(subject + 1) * 340]

    #train_data = np.empty([len(train_balanced_list)*2, 1, channels_n, trial_n], dtype=float)
    train_data = np.empty([len(train_balanced_list), 1, channels_n, trial_n], dtype=float)
    validation_data = np.empty([validation_tot_trials, 1, channels_n, trial_n], dtype=float)
    for index, trial in enumerate(train_balanced_list):
        train_data[index, 0, :, :] = all_data[trial * channels_n:(trial + 1) * channels_n, :trial_n]
        #train_data[index*2, 0, :, :] = all_data[trial * channels_n:(trial + 1) * channels_n, :trial_n]
        #train_data[index*2+1,0,:,:]=all_data[trial * channels_n:(trial + 1) * channels_n, shift:trial_n+shift]
        #print("Loading training data %d" % (index))
    for index, subject in enumerate(validation_list):
        for itera in range(340):
            validation_data[index * 340 + itera, 0, :, :] = all_data[(subject * 340 + itera) * channels_n:(subject * 340 + itera + 1) * channels_n,:trial_n]

    flatten_train_labels = all_labels[train_balanced_list]
    flatten_validation_labels = validation_labels
    train_labels = np_utils.to_categorical(all_labels, 2)
    validation_labels = np_utils.to_categorical(validation_labels, 2)
    train_labels = train_labels[train_balanced_list, :]

#    new_flatten_train_labels = np.empty([len(flatten_train_labels) * 2])
#    for n in range(len(flatten_train_labels)):
#        new_flatten_train_labels[n * 2] = flatten_train_labels[n]
#        new_flatten_train_labels[n * 2 + 1] = flatten_train_labels[n]
#    new_train_labels = np_utils.to_categorical(new_flatten_train_labels, 2)

    if (False):  # load previous model
        model=load_model(input('Model name:')+'.h5')
    else:
        # Define and Train model
        model = EEGNet(nb_classes=2, Chans=channels_n, Samples=trial_n, regRate=0.001, dropoutRate=0.25, kernels=[(2, 32), (8, 4)])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history=model.fit(train_data, train_labels, batch_size=batchsize, epochs=300, verbose=2)
    # model.save('***.h5')
    # generate prediction probabilities
    train_predict = model.predict(train_data, batch_size=batchsize, verbose=0)
    validation_predict = model.predict(validation_data, batch_size=batchsize, verbose=0)
    test_predict = model.predict(test_data, batch_size=batchsize, verbose=0)
    validation_accuracy=model.evaluate(validation_data,validation_labels,batch_size=batchsize)
    print(validation_accuracy)
    test_accuracy=model.evaluate(test_data,test_labels,batch_size=batchsize)
    print(test_accuracy)

    # Calculate AUC
    fpr, tpr, thresholds = roc_curve(flatten_train_labels, train_predict[:, 1], pos_label=1)
    print("Training auc:")
    train_auc_value = auc(fpr, tpr)
    print(train_auc_value)

    fpr, tpr, thresholds = roc_curve(flatten_validation_labels, validation_predict[:, 1], pos_label=1)
    print("Validation auc:")
    validation_auc_value = auc(fpr, tpr)
    print(validation_auc_value)

    fpr, tpr, thresholds = roc_curve(flatten_test_labels, test_predict[:, 1], pos_label=1)
    print("Test auc:")
    test_auc_value = auc(fpr, tpr)
    print(test_auc_value)

    del model

# plot AUC curve of test

lw = 2
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='Test ROC curve (area = %0.2f)' % test_auc_value)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

