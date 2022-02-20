from keras.models import load_model
import numpy as np
import data_load
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score, matthews_corrcoef

def GetMetrics(model, x, y):
    pred = model.predict_classes(x)
    pred_p = model.predict(x)
    fpr, tpr, thresholdTest = roc_curve(y, pred_p)
    aucv = auc(fpr, tpr)
    precision, recall, fscore, support = precision_recall_fscore_support(y, pred, average='macro')
    print('auc,acc,mcc,precision,recall,fscore,support:', aucv, accuracy_score(y, pred), matthews_corrcoef(y, pred),
          precision, recall, fscore, support)
    return [aucv, accuracy_score(y, pred), matthews_corrcoef(y, pred), precision, recall, fscore]

def GetMetrics_IMR90(model, x, y):
    pred = model.predict_classes(x)
    pred_p = model.predict(x)
    print('acc', accuracy_score(y, pred))
    print('number:',pred.sum())
    return [accuracy_score(y, pred)]


x_test = np.load('K562_RF_test.npy')
print(x_test.shape)
y1 = np.ones(int(len(x_test)/2))
y2 = np.zeros(int(len(x_test)/2))
y_test = np.concatenate((y1,y2),axis=0)
print(y_test.shape)

filepath = 'model/K562_RF.hdf5'
GetMetrics(load_model(filepath), x_test, y_test)
'''filepath = 'model/K562.hdf5'
GetMetrics_IMR90(load_model(filepath), x_test, y_test)'''