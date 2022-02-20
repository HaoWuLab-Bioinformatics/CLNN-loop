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

The code "main.py" is used for model training and to evaluate the performance of CLNN-loop on the independent test set. Running "main.py" will import "LSTM.py" automatically.

The file "model" contains the models trained using different cell lines and CBS pair types, which can ensure the reproduction of our experimental results.
## Dependency
Python 3.6   
keras  2.3.1  
sklearn  
numpy  
tensorflow 2.0

## Usage
You should extract features by running the script as follows: 

`python feature_code.py`  

If you want to compile and run CLNN-loop, you can run the script as follows:  

`python main.py`
