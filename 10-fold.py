import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import data_load
import os
import sys
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam, Adadelta
from keras.utils import np_utils
# convolutional layers
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, AveragePooling1D
from keras.layers import Bidirectional
from keras.models import load_model

from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM
from keras.layers import Reshape
from keras.constraints import maxnorm
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report

def GetMetrics(model, x, y):
    pred = model.predict_classes(x)
    pred_p = model.predict(x)
    fpr, tpr, thresholdTest = roc_curve(y, pred_p)
    aucv = auc(fpr, tpr)
    precision, recall, fscore, support = precision_recall_fscore_support(y, pred, average='macro')
    print('auc,acc,mcc,precision,recall,fscore,support:', aucv, accuracy_score(y, pred), matthews_corrcoef(y, pred),
          precision, recall, fscore, support)
    return [aucv, accuracy_score(y, pred), matthews_corrcoef(y, pred), precision, recall, fscore]

def three_CNN_LSTM(x_train, y_train, x_val, y_val, x_test, y_test, learning_rate, INPUT_SHAPE, KERNEL_NUMBER, KERNEL_SIZE, LSTM_UNITS,
                   train_performance, test_performance, i):
    model = Sequential()
    model.add(Conv1D(KERNEL_NUMBER, kernel_size=KERNEL_SIZE, activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling1D())
    model.add(Dropout(0.3))
    model.add(Conv1D(KERNEL_NUMBER, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.3))
    model.add(Conv1D(KERNEL_NUMBER, kernel_size=KERNEL_SIZE, activation='relu'))

    model.add(Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2, recurrent_dropout=0.5)))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    filepath = "model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=6)
    model.compile(loss=LOSS, optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    print(model.summary())
    Tuning = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCH,
                       validation_data=(x_val, y_val), callbacks=[checkpoint, early_stopping_monitor])
    print("train")
    train_performance[i] = GetMetrics(load_model(filepath), x_train, y_train)
    print("test")
    test_performance[i] = GetMetrics(load_model(filepath), x_test, y_test)
    return Tuning, model

path = 'feature/K562/train/RR/'
first = 'left'
second = 'right'
y = np.loadtxt(path + 'RCKmer_'+first+'.txt')[:,0]
feature1_1 = np.loadtxt(path + 'RCKmer_'+first+'.txt')[:,1:]
feature1_2 = np.loadtxt(path + 'RCKmer_'+second+'.txt')[:,1:]
feature2_1 = np.loadtxt(path + 'PCSF_'+first+'.txt')
feature2_2 = np.loadtxt(path + 'PCSF_'+second+'.txt')
x3_1 =np.loadtxt(path + 'PSTNPds_'+first+'.txt')[:,1:]
x3_2 =np.loadtxt(path + 'PSTNPds_'+second+'.txt')[:,1:]
x4_1 =np.loadtxt(path + 'PSTNPss_'+first+'.txt')[:,1:]
x4_2 =np.loadtxt(path + 'PSTNPss_'+second+'.txt')[:,1:]
feature5_1 = np.loadtxt(path + 'NPSE_'+first+'.txt')
feature5_2 = np.loadtxt(path + 'NPSE_'+second+'.txt')
x = data_load.load(feature1_1, feature1_2, feature2_1, feature2_2, x3_1, x3_2, x4_1, x4_2,feature5_1,feature5_2, y)
x = np.expand_dims(x, 2)
x1 = x
y1 = y

path = 'feature/K562/test/RR/'
first = 'left'
second = 'right'
y = np.loadtxt(path + 'RCKmer_'+first+'.txt')[:,0]
feature1_1 = np.loadtxt(path + 'RCKmer_'+first+'.txt')[:,1:]
feature1_2 = np.loadtxt(path + 'RCKmer_'+second+'.txt')[:,1:]
feature2_1 = np.loadtxt(path + 'PCSF_'+first+'.txt')
feature2_2 = np.loadtxt(path + 'PCSF_'+second+'.txt')
x3_1 =np.loadtxt(path + 'PSTNPds_'+first+'.txt')[:,1:]
x3_2 =np.loadtxt(path + 'PSTNPds_'+second+'.txt')[:,1:]
x4_1 =np.loadtxt(path + 'PSTNPss_'+first+'.txt')[:,1:]
x4_2 =np.loadtxt(path + 'PSTNPss_'+second+'.txt')[:,1:]
feature5_1 = np.loadtxt(path + 'NPSE_'+first+'.txt')
feature5_2 = np.loadtxt(path + 'NPSE_'+second+'.txt')
x = data_load.load(feature1_1, feature1_2, feature2_1, feature2_2, x3_1, x3_2, x4_1, x4_2,feature5_1,feature5_2, y)
x = np.expand_dims(x, 2)
x2 = x
y2 = y

x = np.concatenate((x1,x2),axis=0)
y = np.concatenate((y1,y2),axis=0)

INPUT_SHAPE = x.shape[1:3]

LEARNING_RATE = 0.001
KERNEL_NUMBER = 32
KERNEL_SIZE = 5
LSTM_UNITS = 32

MAX_EPOCH = 150
BATCH_SIZE = 50
# VERBOSE = 1
# METRICS =['accuracy']
LOSS = 'binary_crossentropy'

i = 0
acc = np.zeros(10)
aucv = np.zeros(10)
mcc = np.zeros(10)
precision = np.zeros(10)
recall = np.zeros(10)
fscore = np.zeros(10)
train_performance = np.zeros((10,6))
test_performance = np.zeros((10,6))

kf = KFold(10, True, 10)
for train_index, test_index in kf.split(x):
    x_train, x_val, y_train, y_val = train_test_split(x[train_index], y[train_index], test_size=1 / 9, random_state=10)
    x_test = x[test_index]
    y_test = y[test_index]
    three_CNN_LSTM(x_train, y_train, x_val, y_val, x_test, y_test, LEARNING_RATE, INPUT_SHAPE, KERNEL_NUMBER, KERNEL_SIZE,
                        LSTM_UNITS, train_performance, test_performance, i)
    i = i+1
    print(train_performance)
    print(test_performance)

print('auc',train_performance[:,0].mean())
print('acc',train_performance[:,1].mean())
print('mcc',train_performance[:,2].mean())
print('precision',train_performance[:,3].mean())
print('recall',train_performance[:,4].mean())
print('F1 score',train_performance[:,5].mean())


print(train_performance[:,0].mean(),train_performance[:,1].mean(),train_performance[:,2].mean(),
      train_performance[:,3].mean(),train_performance[:,4].mean(),train_performance[:,5].mean())