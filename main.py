# This file contains code required for the preprocessing of data and
# the k-Nearest Neighbours algorithm where k is hyperparameter that will be tuned
# Authors: Jatin Jain, Gavin Williams

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
    # split training data into labels and samples
    explicitLabel = training[[labelname]].to_numpy().reshape(len(training))
    explicitsample = training.drop([labelname], axis=1)

    # train knn
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(explicitsample, explicitLabel)

    # get samples from validation data
    explicitsample = validation.drop([labelname], axis=1)

    # get predictions based on validation samples
    predictions = pd.DataFrame(knc.predict(explicitsample), columns=[labelname])

    # create new array with 1's for each correct prediction and 0's for incorrect
    accuracy = np.where(validation[labelname].reset_index(drop=True) == predictions[labelname], 1, 0)

    # return accuracy and model
    return accuracy.sum()/len(accuracy), knc


# TODO: TUNING
def decisionTree(training, validation, labelname):
    # split training data into labels and samples
    explicitLabel = training[[labelname]].to_numpy().reshape(len(training))
    explicitsample = training.drop([labelname], axis=1)

    # train decision tree
    dtc = DecisionTreeClassifier()
    dtc.fit(explicitsample, explicitLabel)

    # get samples from validation data
    explicitsample = validation.drop([labelname], axis=1)

    # get predictions based on validation samples
    predictions = pd.DataFrame(dtc.predict(explicitsample), columns=[labelname])
    # create new array with 1's for each correct prediction and 0's for incorrect
    accuracy = np.where(validation[labelname].reset_index(drop=True) == predictions[labelname], 1, 0)

    # return accuracy and model
    return accuracy.sum()/len(accuracy), dtc


data = "./archive/data.csv"
cleanedData = preprocess(data)

knnmaxacc = dtcmaxacc = 0
knnbest = dtcbest = None
nfolds = 10
n = len(cleanedData)

for i in range(0, n, int(n/nfolds)):
    j = i+int(n/nfolds)

    # get consecutive entries of proper size
    validationset = cleanedData.iloc[i:j]

    # drop entries in validation ser from training set
    trainingset = cleanedData.drop(validationset.isin(cleanedData).index)

    # train knn and save most accurate model
    accuracy, currmodel = kNN(trainingset, validationset, 'explicit', 10)
    if accuracy > knnmaxacc:
        knnmaxacc = accuracy
        bestknn = currmodel

    # train decision tree and save most accurate model
    accuracy, currmodel = decisionTree(trainingset, validationset, 'mode')
    if accuracy > dtcmaxacc:
        dtcmaxacc = accuracy
        bestdtc = currmodel
