import numpy as np

def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(ranges, (m, 1))
    return normData

def load(feature1_1, feature1_2, feature2_1, feature2_2, x3_1, x3_2, x4_1, x4_2, feature5_1,feature5_2,y):
    feature2_1 = noramlization(feature2_1.reshape(-1, 1))
    feature2_2 = noramlization(feature2_2.reshape(-1, 1))
    feature3_1 = np.zeros(len(y))
    feature3_2 = np.zeros(len(y))
    feature4_1 = np.zeros(len(y))
    feature4_2 = np.zeros(len(y))
    print('load over')
    for i in range(len(y)):
        feature3_1[i] = x3_1[i].sum()
        feature3_2[i] = x3_2[i].sum()
        feature4_1[i] = x4_1[i].sum()
        feature4_2[i] = x4_2[i].sum()
    x = np.concatenate(
        (feature1_1, feature1_2, feature2_1, feature2_2, feature3_1.reshape(-1, 1),
         feature3_2.reshape(-1, 1), feature4_1.reshape(-1, 1), feature4_2.reshape(-1, 1),feature5_1,feature5_2), axis=1)
    return x