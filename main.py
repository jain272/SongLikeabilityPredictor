# This file contains code required for the preprocessing of data and
# the k-Nearest Neighbours algorithm where k is hyperparameter that will be tuned
# Authors: Jatin Jain

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def preprocess(data_csv):
    """Code for preprocessing of data"""
    dfMain = pd.read_csv(data_csv)  # Read data from CSV file into Pandas Dataframe
    dfMain = dfMain[['mode', 'energy', 'acousticness', 'valence', 'explicit', 'danceability',
                     'tempo']]  # Dropping all columns except listed ones
    dfMain = dfMain.reindex(sorted(dfMain.columns), axis=1)  # Sort the columns alphabetically
    dfMain.dropna()  # Dropping rows with missing values
    dfSample = dfMain.sample(n=1000)  # Randomly selecting 1000 rows from the dataframe
    return dfSample


def kNN(training, validation, labelname, k):
    """Code for applying k-Nearest Neighbours algorithm"""
    explicitLabel = training[[labelname]].to_numpy().reshape(len(training))
    explicitsample = training.drop([labelname], axis=1)
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(explicitsample, explicitLabel)
    explicitsample = validation.drop([labelname], axis=1)
    predictions = pd.DataFrame(knc.predict(explicitsample), columns=[labelname])
    accuracy = np.where(validation[labelname].reset_index(drop=True) == predictions[labelname], 1, 0)
    return accuracy.sum()/len(accuracy), knc


def decisionTree(training, validation, labelname):
    explicitLabel = training[[labelname]].to_numpy().reshape(len(training))
    explicitsample = training.drop([labelname], axis=1)
    dtc = DecisionTreeClassifier()
    dtc.fit(explicitsample, explicitLabel)
    explicitsample = validation.drop([labelname], axis=1)
    predictions = pd.DataFrame(dtc.predict(explicitsample), columns=[labelname])
    accuracy = np.where(validation[labelname].reset_index(drop=True) == predictions[labelname], 1, 0)
    return accuracy.sum() / len(accuracy), dtc


data = "./archive/data.csv"
cleanedData = preprocess(data)

knnmaxacc = 0
bestknn = None
dtcmaxacc = 0
bestdtc = None
nfolds = 10
for i in range(0, 1000, 100):
    j = i+int(1000/nfolds)
    validationset = cleanedData.iloc[i:j]
    trainingset = cleanedData.drop(validationset.isin(cleanedData).index)
    accuracy, currmodel = kNN(trainingset, validationset, 'explicit', 10)
    if accuracy > knnmaxacc:
        knnmaxacc = accuracy
        bestknn = currmodel
    accuracy, currmodel = decisionTree(trainingset, validationset, 'mode')
    if accuracy > dtcmaxacc:
        dtcmaxacc = accuracy
        bestdtc = currmodel

