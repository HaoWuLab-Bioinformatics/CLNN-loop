# CLNN-loop
CLNN-loop is a deep learning model to predict chromatin loops in the different cell lines and CTCF-binding sites (CBS) pair types.

## Dataset
The file "data" is the data sets used in this study, which contains two cell lines (K562 and MCF-7) and each cell line contains four types of CBS pair (forward–forward orientation pair, forward–reverse orientation pair, reverse–forward orientation pair and reverse–reverse orientation pair). The naming rules for each file are as follows:

name1_name2_name3_name4.fasta

where name1 represents the cell line of the data, name2 represents the CBS pair type of the data, name3 indicates which sequence of the sequence pair and name4 indicates whether the data is the training set or the independent test set.

The file "dataOfIMR90" is the data sets for evaluating the prediction ability of the CLNN-loop on a new cell lines.

## Overview
 
The code "feature_code.py" is used for feature extraction, including seven feature extraction methods. 

The code "data_load.py" is used for fusing features.

The code "LSTM.py" is used to build the model.

The code "modeltest.py" is used to test our saved models and reproduce experimental results.

The code "main.py" is used for model training and to evaluate the performance of CLNN-loop on the independent test set. Running "main.py" will import "LSTM.py" automatically.

The folder "model" contains the models trained using different cell lines and CBS pair types, which can ensure the reproduction of our experimental results.

The file "K562_RR_train.npy" and the file "K562_RR_test.npy" are the feature files of the training set and test set in the RR orientation of the K562 cell line, which are to ensure that "main.py" can be run directly.
## Dependency
Python 3.6   
keras  2.3.1  
sklearn  
numpy  
tensorflow 2.0

## Usage
You should extract features by running the script as follows: 

`python feature_code.py`  

Note that you need to modify the contents of lines 410, 468 and 558 to extract the features of the data you need. For example, if you want to extract features of the test set for the RR orientation of the K562 cell line, you need modify the contents of lines 410, 468 and 558 as follows:

`filename = 'data/K562/K562_RR_left_test'` (line 410, for specific naming rules, see the above)

`filename = 'data/K562/K562_RR_right_test'` (line 468, for specific naming rules, see the above)

`np.save('K562_RR_test.npy',x)` (line 558, save the extracted features)

If you want to compile and run CLNN-loop, you can run the script as follows:  

`python main.py`
