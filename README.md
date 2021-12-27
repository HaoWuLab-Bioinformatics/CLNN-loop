# CLNN-loop
CLNN-loop is a deep learning model to predict chromatin loops in the different cell lines and CTCF-binding sites (CBS) pair types.

## Dataset
The file "data" is the data sets used in this study, which contains two cell lines (K562 and MCF-7) and each cell line contains four types of CBS pair (forward–forward orientation pair, forward–reverse orientation pair, reverse–forward orientation pair and reverse–reverse orientation pair). The naming rules for each file are as follows:

name1_name2_name3_name4.fasta

where name1 represents the cell line of the data,
the DNA sequences encoded by one-hot matrix. The dataset contains a total of 15057 positive sequences and 15070 negative sequences, and each sequence consists of one thousand bases. We randomly select 80% of the data set as the training set, and the remaining 20% of the data set as the independent testing set using the script as follows:

`x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)`

Finally, the training set contains 12077 positive sequences and 12024 negative sequences, and the independent testing set contains 2980 positive sequences and 3046 negative sequences.

## Overview
 
The code "data prepare.py" is used for data processing.  
The code "feature_code.py" and the code "psednc.py" is used for feature extraction, including seven feature extraction methods. 
The code "model.py" is used for model training and performance evaluation. We perform ten-fold cross-validation on the training set and evaluate the performance of StackTADB on the independent test set.
The file "feature_k-mers_k=6.rar" is the compressed file of the feature used in this study, containing Kmers-based feature (k=6) of training set and independent test set. 

## Dependency
Python 3.6   
keras  2.3.1  
sklearn  
numpy  
mlxtend  
h5py 

## Usage
First, you should perform data preprocessing, you can run the script as: 

`python data prepare.py`  

Then you can extract features you need through running the script as:  

`python feature_code.py` or `python psednc.py`  

Finally if you want to compile and run StackTADB, you can run the script as:  

`python model.py`
