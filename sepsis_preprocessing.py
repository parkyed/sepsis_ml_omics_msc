# install package if necessary

# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# file path for project
file_path = '/Users/Ed/Documents/GitHub/sepsis_ml_omics_msc/scripts/' # CHANGE THIS

# import and check data
raw_data = pd.read_csv(str(file_path)+'genomic_data.csv')

# Eliminate duplicate genes. Testing on various columns, NaN values excluded before looking for duplicates.

print('\nResults of testing for duplicate genes and missing values:\n')
dup_column = ['NuID', 'Search_Key', 'ILMN_Gene', 'RefSeq_ID', 'Entrez_Gene_ID', 'Probe_Id']
for column in dup_column:
    data = raw_data.dropna(subset=[column])
    dup_list = []
    for index, value in data.duplicated(subset=[column]).items():
        if value == True:
            dup_list.append(index)
    print(f"# duplicate rows based on {column}:   "+str(len(dup_list)))

# reformat data, drop duplicates, transpose, index on probe_id, drop columns with all NaN, drop Fold change

gene_df = raw_data.iloc[:, 0:29]
df = raw_data.iloc[:, np.r_[14, 29:93]].drop_duplicates(subset=['Probe_Id'])
df = df.transpose()
df = df.rename(columns=df.iloc[0]).drop(df.index[0])
df = df.drop(['Fold change'], axis=0)
df = df.dropna(axis=1, how='all')

labels = []
for item in df.index:
    if 'Con' in item:
        labels.append(0)
    else:
        labels.append(1)
labels = np.asarray(labels)

bool_series_zeros = (df == 0).all(axis=0)

print('The number of missing values is:   '+str(df.isnull().sum().sum()))
print('The number of columns with all zeros is:   '+str(sum(bool_series_zeros)))

# create copy of the data without patient Inf_075, features and labels

df_no75 = df.drop(['Inf075'], axis=0)

labels_no75 = []
for item in df.index:
    if item == 'Inf075':
        continue
    if 'Con' in item:
        labels_no75.append(0)
    else:
        labels_no75.append(1)
labels_no75 = np.asarray(labels_no75)

# standardise the data - z-scoring

def standardise(examples):
    scaler = StandardScaler()
    examples_scaled = scaler.fit_transform(examples)
    return examples_scaled

# standardised data set including patient Inf_075
X_df = standardise(df)
X_df = pd.DataFrame(X_df, columns=df.columns, index=df.index)

# standardised data set excluding patient Inf_075
X_df_no75 = standardise(df_no75)
X_df_no75 = pd.DataFrame(X_df_no75, columns=df_no75.columns, index=df_no75.index)

# Reduced standardised data set for code testing
X_df_red = X_df.iloc[:, 0:200]

print('\nDimensions of final pre-processed input data sets:\n')
print('Shape of stadardised data set including Inf075:   '+str(X_df.shape))
print('Shape of stadardised data set excluding Inf075:   '+str(X_df_no75.shape))
print('Shape of standardised testing data set:   '+str(X_df_red.shape))

with open(str(file_path)+'scaled_data.pkl', 'wb') as f:
    pickle.dump([X_df, X_df_no75, X_df_red, gene_df, labels, labels_no75], f)
