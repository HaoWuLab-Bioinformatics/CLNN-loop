# CLNN-loop
CLNN-loop is a deep learning model to predict chromatin loops in the different cell lines and CTCF-binding sites (CBS) pair types.

## Dataset
The file "data" is the data sets used in this study, which contains two cell lines (K562 and MCF-7) and each cell line contains four types of CBS pair (forward–forward orientation pair, forward–reverse orientation pair, reverse–forward orientation pair and reverse–reverse orientation pair). The naming rules for each file are as follows:

name1_name2_name3_name4.fasta

where name1 represents the cell line of the data, name2 represents the CBS pair type of the data, name3 indicates which sequence of the sequence pair and name4 indicates whether the data is the training set or the independent test set.

The file "dataOfIMR90" is the data sets for evaluating the prediction ability of the CLNN-loop on a new cell lines.

## Overview
 
The code "feature_code.py" is used for feature extraction, including seven feature extraction methods. 
The code "LSTM.py" is uesd to build the model.
The code "main.py" is used for model training and performance evaluation. We perform ten-fold cross-validation on the training set and evaluate the performance of CLNN-loop on the independent test set. Running "main.py" will import LSTM.py automatically.

## Dependency
Python 3.6   
keras  2.3.1  
sklearn  
numpy  
tensorflow 2.0

## Usage
First, you should perform data preprocessing, you can run the script as: 

`python data prepare.py`  

Then you can extract features you need through running the script as:  

`python feature_code.py` or `python psednc.py`  

Finally if you want to compile and run StackTADB, you can run the script as:  

`python model.py`
