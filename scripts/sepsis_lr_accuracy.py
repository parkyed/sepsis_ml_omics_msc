# install package if necessary
# pip install scikit-optimize

# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression
import skopt
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from math import log
from matplotlib import pyplot
import seaborn as sns
from sklearn.metrics import accuracy_score
from sepsis_evaluation import score_model, mis_class_points, identify_genes_from_coefs
from sepsis_evaluation import lr_results_df_inc_features
import pickle
import warnings
warnings.filterwarnings("ignore", message="The objective has been evaluated at this point before")

# load pre-processed data
file_path = '/Users/Ed/Documents/GitHub/sepsis_ml_omics_msc/scripts/' # CHANGE THIS
with open(str(file_path)+'scaled_data.pkl', 'rb') as f:
    X_df, X_df_no75, X_df_red, gene_df, labels, labels_no75 = pickle.load(f)


# logistic regression model 1: optimising for accuracy of inner cv loop

def nested_cv_lr(df, labels, n_repeats):

    # configure the cross-validation procedure
    cv_outer = RepeatedKFold(n_splits=3, n_repeats=n_repeats)

    # lists to collect scores from each outer fold
    scores_list = []
    best_parameters_list = []
    mis_classified_list = []
    selected_features_list = []

    # create inner loop for hyper parameter optimisation
    for train_ix, test_ix in cv_outer.split(df):

        # split data
        X_train, X_test = df.iloc[train_ix, :], df.iloc[test_ix, :]
        y_train, y_test = labels[train_ix], labels[test_ix]

        model, best_parameter = train_lr_baysian(X_train, y_train)

        # predict on the test set and store the selected features
        y_pred = model.predict(X_test)
        y_pred_probab = model.predict_proba(X_test)

        # evaluate the model
        scores = score_model(y_pred, y_pred_probab, y_test)
        mis_classified = mis_class_points(X_test, y_pred, y_test)
        selected_features = model.coef_[0]

        # add scores for this train test split to lists collecting scores for all splits
        scores_list.append(scores)
        best_parameters_list.append(best_parameter)
        mis_classified_list.append(mis_classified)
        selected_features_list.append(selected_features)

    return scores_list, best_parameters_list, mis_classified_list, selected_features_list

def train_lr_baysian(X_train, y_train):

    # define hyper paramaeter search space
    hyper_p_c = Real(low=1e-6, high=1000.0, prior='log-uniform', name='C')
    search_space_lr = [hyper_p_c]

    # define the objective function to optimise over the hyper-parameter search space. Here we want to minimise 1- the
    # accuracy score - balancing recall and precision in the model.
    @use_named_args(search_space_lr)
    def evaluate_model(**params):
        cv_inner = LeaveOneOut()
        y_val_classes = []
        y_val_predictions = []
        for train_index, val_index in cv_inner.split(X_train):

            X_train_inner, X_val = X_train.iloc[train_index, :], X_train.iloc[val_index, :]
            y_train_inner, y_val = y_train[train_index], y_train[val_index]

            # define the model
            model = LogisticRegression(penalty='l1', solver="liblinear")
            model.set_params(**params)
            model.fit(X_train_inner, y_train_inner)

            y_val_pred = model.predict(X_val)
            y_val_classes.append(y_val)
            y_val_predictions.append(y_val_pred)

        score = accuracy_score(y_val_classes, y_val_predictions)
        return 1.0 - score

    # perform optimization
    result = gp_minimize(evaluate_model, search_space_lr, acq_func="EI", x0=[1.0], n_initial_points=20, n_calls=30)

    # save the best performing value of hyper parameter C
    best_parameter = result.x
    print('Best Parameters: %s' % (result.x))
    print('Best Score: %.3f' % (1.0 - result.fun))

    # retrain the model with fixed value of C
    model = LogisticRegression(C=best_parameter[0],
                                  penalty='l1',
                                  solver="liblinear")
    model.fit(X_train, y_train)

    return model, best_parameter


# run nested cross validation and collect the results
scores_lr, best_parameters_lr, mis_classified_lr, selected_features_list_lr = nested_cv_lr(X_df,  labels, 3)

# create dataframes to present the results
print('\nThe results for logistic regression optimising for accuracy:')
lr_results_df_all_inc75 = lr_results_df_inc_features(scores_lr, best_parameters_lr, mis_classified_lr, selected_features_list_lr)
lr_results_df_all_inc75 = lr_results_df_all_inc75.drop(['False Positives', 'False Negatives'], axis=1)
print(lr_results_df_all_inc75)

# train final model and determine features
final_lr, best_param = train_lr_baysian(X_df, labels)
selected_features_lr = final_lr.coef_[0]
selected_gene_lr_final, selected_gene_lr_df = identify_genes_from_coefs(selected_features_lr, X_df, gene_df)
print('\nThe number of selected genes in the final model is:  '+str(len(selected_gene_lr_final)))
print(selected_gene_lr_df)
