# install package if necessary
# pip install scikit-optimize

# import libraries

from sklearn.model_selection import RepeatedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression
import skopt
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_evaluations
from math import log
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
import sepsis_evaluation

import pickle
import warnings
warnings.filterwarnings("ignore", message="The objective has been evaluated at this point before")

# load pre-processed dataset
file_path = '/Users/Ed/Documents/GitHub/sepsis_ml_omics_msc/' # CHANGE THIS
with open(str(file_path)+'scaled_data.pkl', 'rb') as f:
    X_df, X_df_no75, labels, labels_no75 = pickle.load(f)


# load pre-processed dataset

def nested_cv_lr(df, labels, n_repeats):

    # configure the cross-validation procedure
    cv_outer = RepeatedKFold(n_splits=3, n_repeats=n_repeats)

    # lists to collect scores from each outer fold
    scores_list = []
    best_parameters_list = []
    mis_classified_list = []

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

        # add scores for this train test split to lists collecting scores for all splits
        scores_list.append(scores)
        best_parameters_list.append(best_parameter)
        mis_classified_list.append(mis_classified)

    return scores_list, best_parameters_list, mis_classified_list

def train_lr_baysian(X_train, y_train):
    # define cross validation approach
    cv_inner = LeaveOneOut()

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
    result = gp_minimize(evaluate_model, search_space_lr, n_calls=15)

    # save the best performing value of hyper parameter C
    best_parameter = result.x
    print('Best Parameters: %s' % (result.x))
    print('Best Score: %.3f' % (1.0 - result.fun))
    plot_convergence(result)

    # retrain the model with fixed value of C
    model = LogisticRegression(C=best_parameter[0],
                                  penalty='l1',
                                  solver="liblinear")
    model.fit(X_train, y_train)

    return model, best_parameter

# run nested cross validation on full data set
scores_lr, best_parameters_lr, mis_classified_lr = nested_cv_lr(X_df,  labels, 1)

# run nested cross validation without patient 75
#scores_lr_no75, best_parameters_lr_no75, mis_classified_lr_no75 = nested_cv_lr(X_df_no75, labels_no75, 3)

#print('The results for logistic regression with all 63 examples:')
print(scores_lr)
print(best_parameters_lr)
print(np.mean(np.asarray(scores_lr), axis=0))
print(mis_classified_lr)
