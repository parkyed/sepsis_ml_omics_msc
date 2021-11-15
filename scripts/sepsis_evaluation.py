# import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score


# functions for scoring models
def score_model(y_pred_classes, y_pred_probabilities, y_true_classes):
    accuracy = round(accuracy_score(y_true_classes, y_pred_classes),2)
    recall = round(recall_score(y_true_classes, y_pred_classes),2)
    tn, fp, fn, tp = confusion_matrix(y_true_classes, y_pred_classes).ravel()
    specificity = round((tn / (tn+fp)),2)
    precision = round(precision_score(y_true_classes, y_pred_classes),2)
    auc = round(roc_auc_score(y_true_classes, y_pred_probabilities[:, 1]),2)
    return accuracy, recall, specificity, precision, auc


def mis_class_points(X_test, y_pred_classes, y_true_classes):
    mis_classified = {}
    for row_index, (examples, prediction, label) in enumerate(zip (X_test, y_pred_classes, y_true_classes)):
        if prediction == label:
            continue
        if prediction == 0 and label == 1:
            mis_classified[X_test.iloc[[row_index]].index[0]] = 'fn'
        elif prediction == 1 and label == 0:
            mis_classified[X_test.iloc[[row_index]].index[0]] = 'fp'
    return mis_classified



# functions to create dataframes from model evaluation nResults


# create a dataframe of results, including the number of features per training iteration
def lr_results_df_inc_features(scores, parameters, misclassified, features):
    p_list = []
    for i in parameters:
        p_list.extend(i)
    p_list = np.array(p_list)
    p_list = np.around(p_list, decimals=2)
    p_list = np.append(p_list, ['n/a'], axis=0)
    mean_scores = np.mean(np.asarray(scores), axis=0)
    array = np.array(scores)
    array = np.vstack((array, mean_scores))
    array = pd.DataFrame(array, index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 'Mean'], columns = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'AUC'])
    # FOR TESTING ONLY
    # array = pd.DataFrame(array, index = [1, 2, 3, 'Mean'], columns = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'AUC'])

    array = array.round(2)
    array.insert(0, 'Best C', p_list)

    # create column for the number of features
    feat_count_list = []
    for i in features:
        feat_count = np.count_nonzero(i)
        feat_count_list.append(feat_count)
    feat_count_list = np.append(feat_count_list, [str(round(np.mean(feat_count_list),0))], axis=0)
    feat_count_list = np.array(feat_count_list)
    array.insert(1, 'Num Features', feat_count_list)

    # create columns for false positives and false negatives
    fp_column = []
    fn_column = []
    for i in misclassified:
        fps = []
        fns = []
        for key, val in i.items():
            if val == 'fp':
                fps.append(key)
            if val == 'fn':
                fns.append(key)
        fp_column.append(fps)
        fn_column.append(fns)
    fp_column.append('n/a')
    fn_column.append('n/a')
    array.insert(7, 'False Positives', fp_column)
    array.insert(8, 'False Negatives', fn_column)

    return array


# create a dataframe of results, excluding the number of features per training iteration
def lr_results_df(scores, parameters, misclassified):
    p_list = []
    for i in parameters:
        p_list.extend(i)
    p_list = np.array(p_list)
    p_list = np.around(p_list, decimals=2)
    p_list = np.append(p_list, ['n/a'], axis=0)
    mean_scores = np.mean(np.asarray(scores), axis=0)
    array = np.array(scores)
    array = np.vstack((array, mean_scores))
    array = pd.DataFrame(array, index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 'Mean'], columns = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'AUC'])
    array = array.round(2)
    array.insert(0, 'Best C', p_list)

    # create columns for false positives and false negatives
    fp_column = []
    fn_column = []
    for i in misclassified:
        fps = []
        fns = []
        for key, val in i.items():
            if val == 'fp':
                fps.append(key)
            if val == 'fn':
                fns.append(key)
        fp_column.append(fps)
        fn_column.append(fns)
    fp_column.append('n/a')
    fn_column.append('n/a')
    array.insert(6, 'False Positives', fp_column)
    array.insert(7, 'False Negatives', fn_column)

    return array


# create dataframe from the svm results
def svm_results_df(scores, parameters, misclassified):
    p_list = np.array(parameters)
    p_list = np.around(p_list, decimals=3)
    p_list = np.append(p_list, ['n/a'], axis=0)
    mean_scores = np.mean(np.asarray(scores), axis=0)
    array = np.array(scores)
    array = np.vstack((array, mean_scores))
    array = pd.DataFrame(array, index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 'Mean'], columns = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'AUC'])
    array = array.round(2)
    array.insert(0, 'Best C', p_list)

    # create columns for false positives and false negatives
    fp_column = []
    fn_column = []
    for i in misclassified:
        fps = []
        fns = []
        for key, val in i.items():
            if val == 'fp':
                fps.append(key)
            if val == 'fn':
                fns.append(key)
        fp_column.append(fps)
        fn_column.append(fns)
    fp_column.append('n/a')
    fn_column.append('n/a')
    array.insert(6, 'False Positives', fp_column)
    array.insert(7, 'False Negatives', fn_column)

    return array


# create dataframe from the random forest results
def rf_results_df(scores, parameters, misclassified):
    mean_scores = np.mean(np.asarray(scores), axis=0)
    array = np.array(scores)
    array = np.vstack((array, mean_scores))
    array = pd.DataFrame(array, index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 'Mean'], columns = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'AUC'])
    array = array.round(2)

    # create columns for parameters
    n_est = []
    max_feat = []
    for p in parameters:
        n_est.append(p[0])
        max_feat.append(p[1])
    n_est.append('n/a')
    max_feat.append('n/a')
    array.insert(0, 'Max Features', max_feat)
    array.insert(0, '# Estimators', n_est)


    # create columns for false positives and false negatives
    fp_column = []
    fn_column = []
    for i in misclassified:
        fps = []
        fns = []
        for key, val in i.items():
            if val == 'fp':
                fps.append(key)
            if val == 'fn':
                fns.append(key)
        fp_column.append(fps)
        fn_column.append(fns)
    fp_column.append('n/a')
    fn_column.append('n/a')
    array.insert(7, 'False Positives', fp_column)
    array.insert(8, 'False Negatives', fn_column)

    return array


# functions to identify selected feature names (i.e. genes) from model outputs


# identify genes where model outputs coefficients
def identify_genes_from_coefs(selected_features, dataset, gene_df):
    selected_gene_list = list(zip(dataset.columns, selected_features))

    selected_gene_codes = []
    selected_gene_coefficients = []
    for gene in selected_gene_list:
        if gene[1] != 0:
            selected_gene_codes.append(gene[0])
            selected_gene_coefficients.append(gene[1])

    # identify common gene codes
    selected_gene_df = gene_df[gene_df['Probe_Id'].isin(selected_gene_codes)].iloc[:, np.r_[14, 5]]
    selected_gene_codes_common = list(selected_gene_df.iloc[:,1])

    # zip features importances with common names

    selected_gene_final = list(zip(selected_gene_codes_common, selected_gene_coefficients))
    selected_gene_final = sorted(selected_gene_final, key = lambda x: x[1])

    return selected_gene_final, selected_gene_df



# function to extract genes using feature importances data
def select_genes_fi_pipeline(model, dataset, gene_df):

    # mask of features selected by SelectFromModel
    feature_mask = model.named_steps['select'].get_support()

    # feature importances of the selected genes
    importances = model.named_steps['clf'].feature_importances_

    # extract the selected genes based on the mask
    selected_gene_list = zip(dataset.columns, feature_mask)
    selected_gene_codes = []
    selected_gene_coefficients = []
    for gene in selected_gene_list:
        if gene[1] == True:
            selected_gene_codes.append(gene[0])

    # identify common gene codes
    selected_gene_df = gene_df[gene_df['Probe_Id'].isin(selected_gene_codes)].iloc[:, np.r_[14, 5]]
    selected_gene_codes_common = selected_gene_df.iloc[:,1]

    # combining gene codes with feature importances
    selected_gene_df['FI'] = importances

    # zip features importances with common names
    selected_gene_final = zip(selected_gene_codes_common, importances)
    selected_gene_final = sorted(selected_gene_final, key = lambda x: x[1])

    return selected_gene_final, selected_gene_df



# identify z-scores for the selected genes
def selected_gene_zscores(selected_gene_dataframe, dataset, labels):

    ''' pandas datafram --> pandas datafram
    Function takes a datafrom of the selected genes, with Probe_Id and ILMN_Gene
    and the input dataset of all z-scores across all 48,000 genes, and returns a melded
    dataframe of the selected genes and their Probe_ID names, with z-scores and true class'''

    gene_index_probe = selected_gene_dataframe['Probe_Id'].tolist()
    gene_index_common = selected_gene_dataframe['ILMN_Gene'].tolist()
    gene_labels_dict = dict(zip(gene_index_probe, gene_index_common))
    gene_labels_dict['Class'] = 'Class'
    selected_gene_zscores_df = dataset.loc[:, gene_index_probe]
    selected_gene_zscores_df['Class'] = labels
    selected_gene_zscores_df = selected_gene_zscores_df.rename(columns = gene_labels_dict)
    selected_gene_zscores_df = pd.melt(selected_gene_zscores_df, id_vars=['Class'],
                                                    value_vars=gene_index_common)
    return selected_gene_zscores_df
