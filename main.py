import LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import data_load


path = 'feature/K562/train/FR/'
first = 'F'
second = 'R'
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

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=10)

path = 'feature/K562/test/FF/'
first = 'left'
second = 'right'
y_test1 = np.loadtxt(path + 'RCKmer_'+first+'.txt')[:,0]
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
x = data_load.load(feature1_1, feature1_2, feature2_1, feature2_2, x3_1, x3_2, x4_1, x4_2, feature5_1,feature5_2, y_test1)
x_test1 = np.expand_dims(x, 2)

path = 'feature/K562/test/FR/'
first = 'F'
second = 'R'
y_test2 = np.loadtxt(path + 'RCKmer_'+first+'.txt')[:,0]
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
x = data_load.load(feature1_1, feature1_2, feature2_1, feature2_2, x3_1, x3_2, x4_1, x4_2, feature5_1,feature5_2, y_test2)
x_test2 = np.expand_dims(x, 2)

path = 'feature/K562/test/RF/'
first = 'R'
second = 'F'
y_test3 = np.loadtxt(path + 'RCKmer_'+first+'.txt')[:,0]
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
x = data_load.load(feature1_1, feature1_2, feature2_1, feature2_2, x3_1, x3_2, x4_1, x4_2, feature5_1,feature5_2, y_test3)
x_test3 = np.expand_dims(x, 2)

path = 'feature/K562/test/RR/'
first = 'left'
second = 'right'
y_test4 = np.loadtxt(path + 'RCKmer_'+first+'.txt')[:,0]
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
x = data_load.load(feature1_1, feature1_2, feature2_1, feature2_2, x3_1, x3_2, x4_1, x4_2, feature5_1,feature5_2, y_test4)
x_test4 = np.expand_dims(x, 2)

path = 'feature/MCF7/test/FF/'
first = 'left'
second = 'right'
y_test5 = np.loadtxt(path + 'RCKmer_'+first+'.txt')[:,0]
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
x = data_load.load(feature1_1, feature1_2, feature2_1, feature2_2, x3_1, x3_2, x4_1, x4_2, feature5_1,feature5_2, y_test5)
x_test5 = np.expand_dims(x, 2)

path = 'feature/MCF7/test/FR/'
first = 'F'
second = 'R'
y_test6 = np.loadtxt(path + 'RCKmer_'+first+'.txt')[:,0]
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
x = data_load.load(feature1_1, feature1_2, feature2_1, feature2_2, x3_1, x3_2, x4_1, x4_2, feature5_1,feature5_2, y_test6)
x_test6 = np.expand_dims(x, 2)

path = 'feature/MCF7/test/RF/'
first = 'R'
second = 'F'
y_test7 = np.loadtxt(path + 'RCKmer_'+first+'.txt')[:,0]
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
x = data_load.load(feature1_1, feature1_2, feature2_1, feature2_2, x3_1, x3_2, x4_1, x4_2, feature5_1,feature5_2, y_test7)
x_test7 = np.expand_dims(x, 2)

path = 'feature/MCF7/test/RR/'
first = 'left'
second = 'right'
y_test8 = np.loadtxt(path + 'RCKmer_'+first+'.txt')[:,0]
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
x = data_load.load(feature1_1, feature1_2, feature2_1, feature2_2, x3_1, x3_2, x4_1, x4_2, feature5_1,feature5_2, y_test8)
x_test8 = np.expand_dims(x, 2)


INPUT_SHAPE = x_train.shape[1:3]
'''KERNEL_SIZE = 5
LEARNING_RATE = 0.001
LSTM_UNITS = 32'''

LEARNING_RATE = 0.001
KERNEL_NUMBER = 32
KERNEL_SIZE = 5
LSTM_UNITS = 32



print(x_test.shape)
LSTM.three_CNN_LSTM(x_train, y_train, x_val, y_val,
                    x_test1, y_test1, x_test2, y_test2, x_test3, y_test3, x_test4, y_test4,
                    x_test5, y_test5, x_test6, y_test6, x_test7, y_test7, x_test8, y_test8,
                    LEARNING_RATE, INPUT_SHAPE, KERNEL_SIZE, LSTM_UNITS)
'''LSTM.three_CNN_LSTM1(x_train, y_train, x_val, y_val,
                    x_test, y_test,
                    LEARNING_RATE, INPUT_SHAPE, KERNEL_NUMBER, KERNEL_SIZE, LSTM_UNITS)'''
