# Machine Learning for Sepsis Diagnosis Using Omics Data

Author: Edward Parkinson

This readme accompanies the following files:

1. genomic_data.csv - this file contains the raw gene-expression data for the 63 neonatal infants

2. sepsis_preprocessing.py - code file for the pre-processing steps to transform raw csv into pandas data frames with z-scored data

3. sepsis_evaluation.py - set of functions shared between machine learning models

4. sepsis_lr.py- code to train and evaluate logistic regression model, optimising for the number of output features

5. sepsis_svm.py - code to train and evaluate svm-rfe model

6. sepsis_rf.py - code to train and evaluate the random forest model


To run the analysis and generate the results contained in the dissertation, follow these steps:

- Ensure all the above files are saved in the same folder

- Change the variable named   file_path   in the file sepsis_preprocessing.py to the path of the folder where all files are saved

- Run sepsis_preprocessing.py from the command prompt. This will generate a .pkl file containing the preprocessed data that is used by all three models

- Run each of the model code files in turn from the command prompt (sepsis_lr.py, sepsis_svm.py, and sepsis_rf.py).

Note: the code for each model runs the full model training and evaluation optimising for 30 features in each case. The scritps are designed as an illustration and do not produce all of the results in the dissertation. However, running the scripts to optimise for a different number of features uses the same code. The code for the initial logistic regression model, optimising for accuracy within the inner cross validation loop is not provided  - the code is very similar to the svm case, however the run time is very long.

- The results tables giving the model performance scores are printed to the console

- The final models are optimised for 10 features for logistic regression and 5 features for both SVMs and random forests, as reported in the dissertation

- The graphical illustrations produced are saved to the folder

Note: the analysis relies on multiple randomised train test splits of the data set, so there is no random seed set in the code. The results obtained will vary slightly from those reported in the dissertation.
