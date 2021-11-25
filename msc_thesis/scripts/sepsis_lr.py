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
from sepsis_evaluation import lr_results_df, lr_results_df_inc_features, selected_gene_zscores
import pickle
import warnings
warnings.filterwarnings("ignore", message="The objective has been evaluated at this point before")

# load pre-processed data
file_path = '/Users/Ed/Documents/GitHub/sepsis_ml_omics_msc/scripts/' # CHANGE THIS
with open(str(file_path)+'scaled_data.pkl', 'rb') as f:
    X_df, X_df_no75, X_df_red, gene_df, labels, labels_no75 = pickle.load(f)


# logistic regression model 2: optimising for number of features in inner cv loop

def nested_cv_lr_nf(df, labels, n_repeats, n_features):

    # configure the cross-validation procedure
    cv_outer = RepeatedKFold(n_splits=3, n_repeats=n_repeats)

    # lists to collect scores from each outer fold
    scores_list = []
    best_parameters_list = []
    mis_classified_list = []
    predictions_list = []
    selected_features_list = []

    # create inner loop for hyper parameter optimisation
    for train_ix, test_ix in cv_outer.split(df):

        # split data
        X_train, X_test = df.iloc[train_ix, :], df.iloc[test_ix, :]
        y_train, y_test = labels[train_ix], labels[test_ix]

        model, best_parameter = train_lr_baysian_nf(X_train, y_train, n_features)

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
        predictions_list.append((y_pred, y_pred_probab, y_test))
        selected_features_list.append(selected_features)

    return scores_list, best_parameters_list, mis_classified_list, predictions_list, selected_features_list

# could count the coefficients at each round as well

def train_lr_baysian_nf(X_train, y_train, n_features):

    # define hyper paramaeter search space
    hyper_p_c = Real(low=1e-6, high=1000.0, prior='log-uniform', name='C')
    search_space_lr = [hyper_p_c]
    target_features = n_features

    # define the objective function to optimise over the hyper-parameter search space.
    @use_named_args(search_space_lr)
    def evaluate_model(**params):
        # define the model
        model = LogisticRegression(penalty='l1', solver="liblinear")
        model.set_params(**params)
        model.fit(X_train, y_train)

        n_nonzero = np.sum(model.coef_ != 0)
        return (target_features-n_nonzero)**2

    # perform optimization
    result = gp_minimize(evaluate_model, search_space_lr, acq_func="EI", x0=[1.0], n_initial_points=20, n_calls=30)

    # save the best performing value of hyper parameter C
    best_parameter = result.x
    #print('Best Parameters: %s' % (result.x))

    # retrain the model with fixed value of C
    model = LogisticRegression(C=best_parameter[0],
                                  penalty='l1',
                                  solver="liblinear")
    model.fit(X_train, y_train)

    return model, best_parameter



# run nested cross validation on full data set, 30 features
print("\nResults for all 63 examples optimising for 30 features:\n")
scores_lr_nf, best_parameters_lr_nf, mis_classified_lr_nf, predictions_list_lr_nf, selected_features_list_lr_nf = nested_cv_lr_nf(X_df,  labels, 3, 30)
lr_results_df_nf_inc75 = lr_results_df_inc_features(scores_lr_nf, best_parameters_lr_nf, mis_classified_lr_nf, selected_features_list_lr_nf)
print(lr_results_df_nf_inc75)

# run nested cross validation without patient 75
print("\nResults for examples excluding patient 75, optimising for 30 features:\n")
scores_lr_no75_30f, best_parameters_lr_no75_30f, mis_classified_lr_no75_30f, predictions_list_lr_no75_30f, selected_features_list_lr_no75_30f = nested_cv_lr_nf(X_df_no75, labels_no75, 3, 30)
lr_df_30f = lr_results_df_inc_features(scores_lr_no75_30f, best_parameters_lr_no75_30f, mis_classified_lr_no75_30f, selected_features_list_lr_no75_30f)
print(lr_df_30f)

# training the final model
final_lr_nf, best_param_lr_nf = train_lr_baysian_nf(X_df_no75, labels_no75, 10)
selected_features_lr_nf = final_lr_nf.coef_[0]
selected_gene_lr_nf_final, selected_gene_lr_nf_df = identify_genes_from_coefs(selected_features_lr_nf, X_df, gene_df)
print('\nThe selected features are:\n')
print(selected_gene_lr_nf_df)

# plot z-scores of selected genes
selected_gene_zscores_lr = selected_gene_zscores(selected_gene_lr_nf_df, X_df_no75, labels_no75)

sns.set_style("whitegrid")
lr_gene_plot = sns.catplot(data=selected_gene_zscores_lr,
                           x='variable',
                           y='value',
                           hue='Class',
                           kind='strip',
                           height = 5,
                           aspect = 3,
                           legend=True,
                           legend_out=False,
                           )
lr_gene_plot.set(xlabel='ILMN Gene Name', ylabel='z-scored gene expression level')
lr_gene_plot._legend.set_title('True Diagnosis')
for t, l in zip(lr_gene_plot._legend.texts,("Healthy", "Infected")):
    t.set_text(l)

lr_gene_plot.savefig('lr_gene_plot.png', dpi=400)
