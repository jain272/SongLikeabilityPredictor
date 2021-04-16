# This file contains code required for the preprocessing of data and
# the k-Nearest Neighbours algorithm where k is hyperparameter that will be tuned
# Authors: Jatin Jain

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def preprocess(data_csv):
    """Code for preprocessing of data"""
    dfMain = pd.read_csv(data_csv)  # Read data from CSV file into Pandas Dataframe
    dfMain = dfMain[['mode', 'energy', 'acousticness', 'valence', 'explicit', 'danceability',
                     'tempo']]  # Dropping all columns except listed ones
    dfMain = dfMain.reindex(sorted(dfMain.columns), axis=1)  # Sort the columns alphabetically
    dfMain.dropna()  # Dropping rows with missing values
    dfSample = dfMain.sample(n=1000)  # Randomly selecting 1000 rows from the dataframe
    return dfSample


def kNN(training, validation, k):
    """Code for applying k-Nearest Neighbours algorithm"""
    explicitLabel = training[['explicit']].to_numpy().reshape(len(training))
    explicitsample = training.drop(['explicit'], axis=1)
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(explicitsample, explicitLabel)
    explicitsample = validation.drop(['explicit'], axis=1)
    predictions = pd.DataFrame(neigh.predict(explicitsample), columns=['explicit'])
    accuracy = np.where(validation['explicit'].reset_index(drop=True) == predictions['explicit'], 1, 0)
    return accuracy.sum()/len(accuracy), neigh


data = "./archive/data.csv"
cleanedData = preprocess(data)

maxacc = 0
bestknn = None
nfolds = 10
for i in range(0, 1000, 100):
    j = i+int(1000/nfolds)
    validationset = cleanedData.iloc[i:j]
    trainingset = cleanedData.drop(validationset.isin(cleanedData).index)
    accuracy, knnmodel = kNN(trainingset, validationset, 10)
    print(accuracy)
    if accuracy > maxacc:
        bestknn = maxacc
"""
modelabel = cleanedData[['mode']]
modesample = cleanedData.drop(['mode'], axis=1)"""

