# install package if necessary
# pip install scikit-optimize

# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, LeaveOneOut, ParameterGrid
from sklearn.feature_selection import mutual_info_classif, SelectKBest, RFE
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import skopt
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from math import log
from matplotlib import pyplot
import seaborn as sns
from sklearn.metrics import accuracy_score
from sepsis_evaluation import score_model, mis_class_points
from sepsis_evaluation import svm_results_df, selected_gene_zscores
import pickle
import warnings
warnings.filterwarnings("ignore", message="The objective has been evaluated at this point before")

# load pre-processed data
file_path = '/Users/Ed/Documents/GitHub/sepsis_ml_omics_msc/scripts/' # CHANGE THIS
with open(str(file_path)+'scaled_data.pkl', 'rb') as f:
    X_df, X_df_no75, X_df_red, gene_df, labels, labels_no75 = pickle.load(f)


# support vector machines with recursive feature elimination

# functions to implement univariate feature selection based on kbest, to n_features

def filter_features(X_train, y_train, X_test, n_features):
    fs=SelectKBest(mutual_info_classif, k=n_features).fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    gene_list = list(zip(X_train.columns, fs.get_support()))
    new_features = []
    for gene in gene_list:
        if gene[1] == True:
            new_features.append(gene[0])
    X_train = pd.DataFrame(X_train_fs, columns=new_features, index=X_train.index)
    X_test = pd.DataFrame(X_test_fs, columns=new_features, index=X_test.index)
    return X_train, X_test, new_features

def filter_features_final(X_train, y_train, n_features):
    fs=SelectKBest(mutual_info_classif, k=n_features).fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    gene_list = list(zip(X_train.columns, fs.get_support()))
    new_features = []
    for gene in gene_list:
        if gene[1] == True:
            new_features.append(gene[0])
    X_train = pd.DataFrame(X_train_fs, columns=new_features, index=X_train.index)
    return X_train, new_features

# nexted cross validation function for SVM

def nested_cv_svm_baysian(df, labels, n_repeats, in_features, out_features):
    '''dataframe, labels, integer -> lists of scores, best hyper parameters, selected features, mis-classified points

    This function implements a nested cross validation. The outer loop is a repeated k fold, that creates multiple
    alternative train / test splits. The inner loop performs hyper parameter optimisation and feature selection using
    recursive feature elimination using an SVM model.
    '''

    # configure the cross-validation procedure
    cv_outer = RepeatedKFold(n_splits=3, n_repeats=n_repeats)

    # lists to collect scores from each outer fold
    scores_list = []
    best_parameters_list = []
    mis_classified_list = []
    filtered_features_list = []
    selected_features_list = []

    # create inner loop for hyper parameter optimisation
    for train_ix, test_ix in cv_outer.split(df):

        # split data
        X_train, X_test = df.iloc[train_ix, :], df.iloc[test_ix, :]
        y_train, y_test = labels[train_ix], labels[test_ix]

        # implement univariate feature selection
        X_train, X_test, new_features = filter_features(X_train, y_train, X_test, in_features)

        # train and fit the model
        model, best_C = train_svm_baysian(X_train, y_train, out_features)

        # model predictions on the test set
        y_pred = model.predict(X_test)
        y_pred_probab = model.predict_proba(X_test)

        # evaluate the model
        scores = score_model(y_pred, y_pred_probab, y_test)
        mis_classified = mis_class_points(X_test, y_pred, y_test)
        selected_features = model.named_steps['feature_elim'].support_

        # add scores to lists
        scores_list.append(scores)
        best_parameters_list.append(best_C)
        mis_classified_list.append(mis_classified)
        filtered_features_list.append(new_features)
        selected_features_list.append(selected_features)

    return scores_list, best_parameters_list, mis_classified_list, filtered_features_list, selected_features_list #added 4+5 param

# Baysian optimisation with Gaussian processes for SVM hyper parameter optimisation

def train_svm_baysian(X_train, y_train, out_features):

    # define hyper paramaeter search space
    hyper_p_c = Real(1e-6, 100.0, 'log-uniform', name='C')
    search_space_svm = [hyper_p_c]

    # define the objective function to optimise
    @use_named_args(search_space_svm)
    def evaluate_model(**params):
        cv_inner = LeaveOneOut()
        y_val_classes = []
        y_val_predictions = []
        for train_index, val_index in cv_inner.split(X_train):

            X_train_inner, X_val = X_train.iloc[train_index, :], X_train.iloc[val_index, :]
            y_train_inner, y_val = y_train[train_index], y_train[val_index]

            # define the pipeline
            svc = SVC(kernel="linear", probability=True)
            svc.set_params(**params)
            rfe = RFE(estimator=svc, n_features_to_select=out_features, step=0.1)
            svm_pipeline = Pipeline(steps=[('rfe', rfe),
                                           ('model', svc)])

            svm_pipeline.fit(X_train_inner, y_train_inner)

            y_val_pred = svm_pipeline.predict(X_val)
            y_val_classes.append(y_val)
            y_val_predictions.append(y_val_pred)

        score = accuracy_score(y_val_classes, y_val_predictions)
        return 1.0 - score

    # perform gp optimization, and store best hyper parameters
    result = gp_minimize(evaluate_model, search_space_svm, n_calls=20)
    best_C = result.x[0]

    # retrain the model with selected hyper parameters
    clf = SVC(kernel="linear", C=best_C, probability=True)
    feature_elim = RFE(estimator=clf, n_features_to_select=out_features, step=0.1)
    model = Pipeline(steps=[('feature_elim', feature_elim),
                            ('clf', clf)])

    model.fit(X_train, y_train)

    return model, best_C


# train on the no75 dataset optimising for 30 features
print('\nResults for examples excluding patient 75, optimising for 30 features:\n')
scores_svm_no75_b30, best_parameters_svm_no75_b30, mis_classified_svm_no75_b30, filtered_features_svm_no75_b30, selected_features_svm_no75_b30 = nested_cv_svm_baysian(X_df_no75, labels_no75, 3, 250, 30)
svm_df_b30 = svm_results_df(scores_svm_no75_b30, best_parameters_svm_no75_b30, mis_classified_svm_no75_b30)
print(svm_df_b30)

# train final model
X_df_filtered_svm_b, filtered_gene_codes_svm_b = filter_features_final(X_df_no75, labels_no75, 250)
final_svm_b, best_C_b = train_svm_baysian(X_df_filtered_svm_b, labels_no75, 5)

# identify the selected features
selected_features_svm_b = final_svm_b.named_steps['feature_elim'].support_
selected_gene_list_svm_b = zip(filtered_gene_codes_svm_b, selected_features_svm_b)
selected_gene_codes_svm_b = []
for gene in selected_gene_list_svm_b:
    if gene[1] == True:
        selected_gene_codes_svm_b.append(gene[0])

# identify common gene codes
selected_gene_svm_b_df = gene_df[gene_df['Probe_Id'].isin(selected_gene_codes_svm_b)].iloc[:, np.r_[14, 5]]
print('\nThe selected features are:\n')
print(selected_gene_svm_b_df)

# plot z-scores of selected genes
selected_gene_zscores_svm = selected_gene_zscores(selected_gene_svm_b_df, X_df_no75, labels_no75)
sns.set_style("whitegrid")
svm_gene_plot = sns.catplot(data=selected_gene_zscores_svm,
                           x='variable',
                           y='value',
                           hue='Class',
                           kind='strip',
                           height = 5,
                           aspect = 3,
                           legend=True,
                           legend_out=False,
                           )
svm_gene_plot.set(xlabel='ILMN Gene Name', ylabel='z-scored gene expression level')
svm_gene_plot._legend.set_title('True Diagnosis')
for t, l in zip(svm_gene_plot._legend.texts,("Healthy", "Infected")):
    t.set_text(l)

svm_gene_plot.savefig('svm_gene_plot.png', dpi=400)
