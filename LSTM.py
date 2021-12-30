import os
import sys
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score
import numpy as np
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

from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import Reshape
from keras.constraints import maxnorm
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report

MAX_EPOCH = 150
BATCH_SIZE = 50
# VERBOSE = 1
# METRICS =['accuracy']
LOSS = 'binary_crossentropy'


def SaveHistory(Tuning, outfile):
    Hist = np.empty(shape=(len(Tuning.history['val_loss']), 4))
    Hist[:, 0] = Tuning.history['val_loss']
    Hist[:, 1] = Tuning.history['val_accuracy']
    Hist[:, 2] = Tuning.history['loss']
    Hist[:, 3] = Tuning.history['accuracy']
    np.savetxt(outfile, Hist, fmt='%.8f', delimiter=",", header="val_loss,val_acc,train_loss,train_acc", comments="")
    return Hist


def GetMetrics(model, x, y):
    pred = model.predict_classes(x)
    pred_p = model.predict(x)
    fpr, tpr, thresholdTest = roc_curve(y, pred_p)
    aucv = auc(fpr, tpr)
    precision, recall, fscore, support = precision_recall_fscore_support(y, pred, average='macro')
    print('auc,acc,mcc,precision,recall,fscore,support:', aucv, accuracy_score(y, pred), matthews_corrcoef(y, pred),
          precision, recall, fscore, support)
    return [aucv, accuracy_score(y, pred), matthews_corrcoef(y, pred), precision, recall, fscore]


def three_CNN_LSTM(x_train, y_train, x_val, y_val,
                   x_test1, y_test1, x_test2, y_test2, x_test3, y_test3, x_test4, y_test4,
                   x_test5, y_test5, x_test6, y_test6, x_test7, y_test7, x_test8, y_test8,
                   learning_rate, INPUT_SHAPE, KERNEL_SIZE, LSTM_UNITS,):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=KERNEL_SIZE, activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling1D())
    model.add(Dropout(0.3))
    model.add(Conv1D(32, kernel_size=KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.3))
    model.add(Conv1D(32, kernel_size=KERNEL_SIZE, activation='relu'))
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
    # train_performance[i] = GetMetrics(load_model(filepath), x_train, y_train)
    GetMetrics(load_model(filepath), x_train, y_train)
    print("test")
    # test_performance[i] = GetMetrics(load_model(filepath), x_test, y_test)
    print('K562_FF')
    GetMetrics(load_model(filepath), x_test1, y_test1)
    print('K562_FR')
    GetMetrics(load_model(filepath), x_test2, y_test2)
    print('K562_RF')
    GetMetrics(load_model(filepath), x_test3, y_test3)
    print('K562_RR')
    GetMetrics(load_model(filepath), x_test4, y_test4)
    print('MCF7_FF')
    GetMetrics(load_model(filepath), x_test5, y_test5)
    print('MCF7_FR')
    GetMetrics(load_model(filepath), x_test6, y_test6)
    print('MCF7_RF')
    GetMetrics(load_model(filepath), x_test7, y_test7)
    print('MCF7_RR')
    GetMetrics(load_model(filepath), x_test8, y_test8)

    SaveHistory(Tuning, "model.txt")
    return Tuning, model


def three_CNN_LSTM1(x_train, y_train, x_val, y_val, x_test, y_test, learning_rate, INPUT_SHAPE, KERNEL_NUMBER, KERNEL_SIZE, LSTM_UNITS):
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

    filepath = "best_two_CNN_LSTM.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=6)
    model.compile(loss=LOSS, optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    print(model.summary())
    Tuning = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCH,
                       validation_data=(x_val, y_val), callbacks=[checkpoint, early_stopping_monitor])
    print("train")
    train_performance = GetMetrics(load_model(filepath), x_train, y_train)
    print("test")
    test_performance = GetMetrics(load_model(filepath), x_test, y_test)
    SaveHistory(Tuning, "best_two_CNN_LSTM.txt")
    return Tuning, model