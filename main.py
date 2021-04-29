# This file contains code required for the preprocessing of data and
# the k-Nearest Neighbours algorithm where k is hyperparameter that will be tuned
# Authors: Jatin Jain, Gavin Williams

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def trainkNN(training, labelname, k):
    # split training data into labels and samples
    explicitLabel = training[[labelname]].to_numpy().reshape(len(training))
    explicitsample = training.drop([labelname], axis=1)

    # train knn
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(explicitsample, explicitLabel)

    # return accuracy and model
    return knc


# TODO: TUNING
def trainDecisionTree(training, labelname):
    # split training data into labels and samples
    explicitLabel = training[[labelname]].to_numpy().reshape(len(training))
    explicitsample = training.drop([labelname], axis=1)

    # train decision tree
    dtc = DecisionTreeClassifier()
    dtc.fit(explicitsample, explicitLabel)

    # return model
    return dtc


def validate(model, validation, labelname):
    # get samples from validation data
    explicitsample = validation.drop([labelname], axis=1)

    # get predictions based on validation samples
    predictions = pd.DataFrame(model.predict(explicitsample), columns=[labelname])

    # create new array with 1's for each correct prediction and 0's for incorrect
    accuracy = np.where(validation[labelname].reset_index(drop=True) == predictions[labelname], 1, 0)

    return accuracy.sum() / len(accuracy)


def samplevsaccuracy(cleanedData):
    numsampleslistknn = []  # list with number of samples for KNN
    accuracylistknn = []  # list with corresponding accuracies for number of samples for KNN
    numsampleslistdtc = []  # list with number of samples for Decision Trees
    accuracylistdtc = []  # list with corresponding accuracies for number of samples for Decision Trees

    fixedvalset = cleanedData.sample(n=100)  # Get 100 random cleaned samples with labels

    # Drop the validation data points from total data
    trainingsetall = cleanedData.drop(fixedvalset.isin(cleanedData).index)

    for count in range(1, 10):
        # Get data points for training, increasing each time by 100 samples
        trainingset = trainingsetall.sample(n=(count * 100))

        # Train the KNN and Decision Tree on the training set for current iteration
        bestknn, bestdtc = train(trainingset)

        # Obtain accuracy for knn classifier on the given dataset
        accuracyknn = validate(bestknn, fixedvalset, 'explicit')
        accuracydtc = validate(bestdtc, fixedvalset, 'mode')

        # Append the number of samples and corresponding accuracies to the corresponding lists
        numsampleslistknn.append(len(trainingset))
        numsampleslistdtc.append(len(trainingset))
        accuracylistknn.append(accuracyknn)
        accuracylistdtc.append(accuracydtc)

    # Plot corresponding lists for the accuracy vs sample plots
    plt.plot(numsampleslistknn, accuracylistknn)
    plt.xlabel('Number of Samples')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Sample Plot for K-Nearest Neighbors')
    plt.show()
    plt.plot(numsampleslistdtc, accuracylistdtc)
    plt.xlabel('Number of Samples')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Sample Plot for Decision Trees')
    plt.show()


def train(cleanedData):
    knnlabel = 'explicit'
    dtclabel = 'mode'
    knnmaxacc = dtcmaxacc = 0
    bestknn = bestdtc = None
    nfolds = 10
    n = len(cleanedData)

    for i in range(0, n, int(n / nfolds)):
        j = i + int(n / nfolds)

        # get consecutive entries of proper size
        validationset = cleanedData.iloc[i:j]

        # drop entries in validation ser from training set
        trainingset = cleanedData.drop(validationset.isin(cleanedData).index)

        # train knn and save most accurate model
        currmodel = trainkNN(trainingset, knnlabel, 5)
        accuracy = validate(currmodel, validationset, knnlabel)
        if accuracy > knnmaxacc:
            knnmaxacc = accuracy
            bestknn = currmodel

        # train decision tree and save most accurate model
        currmodel = trainDecisionTree(trainingset, dtclabel)
        accuracy = validate(currmodel, validationset, dtclabel)
        if accuracy > dtcmaxacc:
            dtcmaxacc = accuracy
            bestdtc = currmodel
    return bestknn, bestdtc


data = "./archive/data.csv"
cleanedData = preprocess(data)
knn, dtc = train(cleanedData)

testdata = preprocess(data)
print('dtc')
print(validate(dtc, testdata, 'mode'))
print('knn')
print(validate(knn, testdata, 'explicit'))

samplevsaccuracy(cleanedData)
