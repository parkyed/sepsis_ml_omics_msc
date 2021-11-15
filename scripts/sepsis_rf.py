# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, LeaveOneOut, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
import seaborn as sns
from sklearn.metrics import accuracy_score
from sepsis_evaluation import score_model, mis_class_points
from sepsis_evaluation import rf_results_df, selected_gene_zscores, select_genes_fi_pipeline
import pickle
import warnings
warnings.filterwarnings("ignore", message="The objective has been evaluated at this point before")

# load pre-processed data
file_path = '/Users/Ed/Documents/GitHub/sepsis_ml_omics_msc/scripts/' # CHANGE THIS
with open(str(file_path)+'scaled_data.pkl', 'rb') as f:
    X_df, X_df_no75, X_df_red, gene_df, labels, labels_no75 = pickle.load(f)


# nested cross validation functions for random forest model

def nested_cv_rf_oob(df, labels, n_repeats, n_features):
    '''dataframe, labels, integer -> lists of scores, best hyper parameters, selected features, mis-classified points

    This function implements a nested cross validation. The outer loop is a repeated k fold, that creates multiple
    alternative train / test splits. The inner loop performs hyper parameter optimisation and feature selection using
    using by scoring out of bag examples.
    '''

    # configure the cross-validation procedure
    cv_outer = RepeatedKFold(n_splits=3, n_repeats=n_repeats)

    # lists to collect scores from each outer fold
    scores_list = []
    best_parameters_list = []
    mis_classified_list = []
    selected_features_list = []           # added this line

    # create inner loop for hyper parameter optimisation
    for train_ix, test_ix in cv_outer.split(df):

        # split data
        X_train, X_test = df.iloc[train_ix, :], df.iloc[test_ix, :]
        y_train, y_test = labels[train_ix], labels[test_ix]

        # train and fit the model
        model, best_n_estimators, best_max_features, selected_features = train_rf_oob(X_train, y_train, n_features)  # changed selected_features

        # model predictions on the test set
        y_pred = model.predict(X_test)
        y_pred_probab = model.predict_proba(X_test)

        # evaluate the model
        scores = score_model(y_pred, y_pred_probab, y_test)
        mis_classified = mis_class_points(X_test, y_pred, y_test)

        # add scores to lists
        scores_list.append(scores)
        best_parameters_list.append((best_n_estimators, best_max_features))
        mis_classified_list.append(mis_classified)
        selected_features_list.append(selected_features)            # added this line

    return scores_list, best_parameters_list, mis_classified_list, selected_features_list


def train_rf_oob(X_train, y_train, n_features):

    # define pipeline and perform search
    n_estimators = [int(x) for x in [500, 1000]]  # n_estimators large since low sample size
    max_features = [100, 200, 500]
    param_grid = {'n_estimators': n_estimators,
                   'max_features': max_features
                   }

    cv_res= []

    for param in ParameterGrid(param_grid):

        # fit model
        rf_clf_select = RandomForestClassifier()
        rf_clf_select.set_params(**param)
        rf_clf = RandomForestClassifier(n_estimators=1000, max_features=None, oob_score = True)
        rf_select = SelectFromModel(rf_clf_select, max_features=n_features, threshold=-np.inf)
        pipe = Pipeline(steps=[('select', rf_select),
                               ('clf', rf_clf)])

        pipe.fit(X_train, y_train)

        score = pipe.named_steps['clf'].oob_score_
        #print('Parameters:'+str(param)+'  Score:'+str(score))
        cv_res.append([param['n_estimators'], param['max_features'], score])

    cv_res = np.asarray(cv_res)
    best_n_estimators = int(cv_res[np.argmax(cv_res[:,2])][0])
    best_max_features = int(cv_res[np.argmax(cv_res[:,2])][1])

    # retrain the model with optimum hyper parameters
    rf_clf_select = RandomForestClassifier(n_estimators=best_n_estimators, max_features=best_max_features)
    rf_clf = RandomForestClassifier(n_estimators=best_n_estimators, max_features=None, oob_score = True)
    rf_select = SelectFromModel(rf_clf_select, max_features=n_features, threshold=-np.inf)
    model = Pipeline(steps=[('select', rf_select),
                            ('clf', rf_clf)])
    model.fit(X_train, y_train)
    selected_features = model.named_steps['select'].get_support()

    #print('Best no. estimators:  '+str(best_n_estimators)+
    #      ',    Best max features:  '+str(best_max_features)+
    #      ',    Number of features:  '+str(n_features))

    return model, best_n_estimators, best_max_features, selected_features


# train on the no75 dataset optimising for 30 features
print('\nResults for examples excluding patient 75, optimising for 30 features:\n')
scores_rf_no75_oob30, best_parameters_rf_no75_oob30, mis_classified_rf_no75_oob30, selected_features_rf_no75_oob30 = nested_cv_rf_oob(X_df_no75, labels_no75, 3, 30)
rf_df_30 = rf_results_df(scores_rf_no75_oob30, best_parameters_rf_no75_oob30, mis_classified_rf_no75_oob30)
rf_df_30_drophyper = rf_df_30.drop(['# Estimators', 'Max Features'], axis=1)
print(rf_df_30_drophyper)

# train final model
final_rf_oob, best_n_estimators_oob, best_max_features_oob, rf_selected_features_oob = train_rf_oob(X_df_no75, labels_no75, 5)

# identify the selected features
print('\nThe selected features are:\n')
selected_gene_rf_final_oob, selected_gene_rf_df = select_genes_fi_pipeline(final_rf_oob, X_df_no75, gene_df)
print(selected_gene_rf_df)

# plot z-scores of selected genes
selected_gene_zscores_rf = selected_gene_zscores(selected_gene_rf_df, X_df_no75, labels_no75)
sns.set_style("whitegrid")
rf_gene_plot = sns.catplot(data=selected_gene_zscores_rf,
                           x='variable',
                           y='value',
                           hue='Class',
                           kind='strip',
                           height = 5,
                           aspect = 3,
                           legend=True,
                           legend_out=False,
                           )
rf_gene_plot.set(xlabel='ILMN Gene Name', ylabel='z-scored gene expression level')
rf_gene_plot._legend.set_title('True Diagnosis')
for t, l in zip(rf_gene_plot._legend.texts,("Healthy", "Infected")):
    t.set_text(l)

rf_gene_plot.savefig('rf_gene_plot.png', dpi=400)
