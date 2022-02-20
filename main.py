import LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import data_load


x = np.load('K562_RF_train.npy')
print(x.shape)
y1 = np.ones(int(len(x)/2))
y2 = np.zeros(int(len(x)/2))
y = np.concatenate((y1,y2),axis=0)
print(y.shape)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=10)

x_test = np.load('K562_RF_test.npy')
print(x_test.shape)
y1 = np.ones(int(len(x_test)/2))
y2 = np.zeros(int(len(x_test)/2))
y_test = np.concatenate((y1,y2),axis=0)
print(y_test.shape)


INPUT_SHAPE = x_train.shape[1:3]
'''KERNEL_SIZE = 5
LEARNING_RATE = 0.001
LSTM_UNITS = 32'''

LEARNING_RATE = 0.001
KERNEL_NUMBER = 32
KERNEL_SIZE = 5
LSTM_UNITS = 32


print(x_test.shape)
LSTM.three_CNN_LSTM1(x_train, y_train, x_val, y_val,
                    x_test, y_test, LEARNING_RATE, INPUT_SHAPE, KERNEL_NUMBER, KERNEL_SIZE, LSTM_UNITS)
'''LSTM.three_CNN_LSTM1(x_train, y_train, x_val, y_val,
                    x_test, y_test,
                    LEARNING_RATE, INPUT_SHAPE, KERNEL_NUMBER, KERNEL_SIZE, LSTM_UNITS)'''
