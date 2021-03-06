# This file contains code required for the preprocessing of data and
# the k-Nearest Neighbours algorithm where k is hyperparameter that will be tuned
# It also contains the training of the decision tree classifier and a function
# that calculates the accuracy of the trained classifier and a function that can plot the
# Accuracy vs. Sample Plots for both the classifiers.
# Authors: Jatin Jain, Gavin Williams

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def preprocess(data_csv, samples):
    """Code for preprocessing of data"""
    dfMain = pd.read_csv(data_csv)  # Read data from CSV file into Pandas Dataframe
    dfMain = dfMain[['mode', 'energy', 'acousticness', 'valence', 'explicit', 'danceability',
                     'tempo']]  # Dropping all columns except listed ones
    dfMain = dfMain.reindex(sorted(dfMain.columns), axis=1)  # Sort the columns alphabetically
    dfMain.dropna()  # Dropping rows with missing values
    dfSample = dfMain.sample(n=samples)  # Randomly selecting 1000 rows from the dataframe
    return dfSample


def trainkNN(training, labelname, k):
    """Code for training the K-Nearest Neighbors classifier"""
    # split training data into labels and samples
    explicitLabel = training[[labelname]].to_numpy().reshape(len(training))
    explicitsample = training.drop([labelname], axis=1)

    # train knn
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(explicitsample, explicitLabel)

    # return accuracy and model
    return knc


# TODO: TUNING
def trainDTC(training, labelname, depth):
    """Code for training the Decision Tree classifier"""
    # split training data into labels and samples
    explicitLabel = training[[labelname]].to_numpy().reshape(len(training))
    explicitsample = training.drop([labelname], axis=1)

    # train decision tree
    dtc = DecisionTreeClassifier(max_depth=depth)
    dtc.fit(explicitsample, explicitLabel)

    # return model
    return dtc


def validate(model, validation, labelname):
    """Code for obtaining the accuracy of any trained classifier"""
    # get samples from validation data
    explicitsample = validation.drop([labelname], axis=1)

    # get predictions based on validation samples
    predictions = pd.DataFrame(model.predict(explicitsample), columns=[labelname])

    # create new array with 1's for each correct prediction and 0's for incorrect
    accuracy = np.where(validation[labelname].reset_index(drop=True) == predictions[labelname], 1, 0)

    return accuracy.sum() / len(accuracy)


def samplevsaccuracy(cleanedData, nfolds, knnlabel, dtclabel):
    """Code for plotting the Accuracy vs Sample Plots for both classifiers"""
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
        bestknn, bestdtc = train(trainingset, nfolds, knnlabel, dtclabel)

        # Obtain accuracy for knn classifier on the given dataset
        accuracyknn = validate(bestknn, fixedvalset, 'explicit')
        accuracydtc = validate(bestdtc, fixedvalset, 'mode')

        # Append the number of samples and corresponding accuracies to the corresponding lists
        numsampleslistknn.append(len(trainingset))
        numsampleslistdtc.append(len(trainingset))
        accuracylistknn.append(accuracyknn)
        accuracylistdtc.append(accuracydtc)

    # Plot corresponding lists for the accuracy vs sample plots
    plot(numsampleslistknn, accuracylistknn, 'Number of Samples', 'Accuracy', 'Accuracy vs Sample Plot for KNN', False)
    plot(numsampleslistdtc, accuracylistdtc, 'Number of Samples', 'Accuracy', 'Accuracy vs Sample Plot for DTC', False)


# plotting helper function
def plot(x, y, xlab, ylab, title, point):
    if point:
        plt.plot(x, y, 'ro')
    else:
        plt.plot(x, y)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.show()


def ROC(validation, predictions, label, threshold):
    # get indices which fall above threshold
    predictions = pd.DataFrame(np.where(predictions[1] >= threshold, 1, 0), columns=[label])

    # true positives, true negatives, false positives, false negatives
    tp = np.where((validation[label].reset_index(drop=True) == 1) & (predictions[label] == 1), 1, 0).sum()
    tn = np.where((validation[label].reset_index(drop=True) == 0) & (predictions[label] == 0), 1, 0).sum()
    fp = np.where((validation[label].reset_index(drop=True) == 0) & (predictions[label] == 1), 1, 0).sum()
    fn = np.where((validation[label].reset_index(drop=True) == 1) & (predictions[label] == 0), 1, 0).sum()

    # true positive rate
    tpr = (tp+1)/(tp+fn+1)

    # false positive rate
    fpr = (fp+1)/(fp+tn+1)

    return tpr, fpr


def ROCplot(actual, predicted, label, thresholds):
    rtpr = []
    rfpr = []

    for threshold in thresholds:
        tpr, fpr = ROC(actual, predicted, label, threshold)
        rtpr.append(tpr)
        rfpr.append(fpr)

    return rtpr, rfpr


def train(cleanedData, nfolds, knnlabel, dtclabel):
    """Umbrella function for the training process"""
    knnmaxacc = dtcmaxacc = 0
    bestknn = bestdtc = None

    n = len(cleanedData)

    for i in range(0, n, int(n / nfolds)):
        j = i + int(n / nfolds)

        # get consecutive entries of proper size
        validationset = cleanedData.iloc[i:j]

        # drop entries in validation ser from training set
        trainingset = cleanedData.drop(validationset.isin(cleanedData).index)

        # tune K, begin at 5 and go to 50 in increments of 5
        for c in range(1, nfolds+1):
            # train knn and save most accurate model
            currmodel = trainkNN(trainingset, knnlabel, 5*c)
            accuracy = validate(currmodel, validationset, knnlabel)
            if accuracy > knnmaxacc:
                knnmaxacc = accuracy
                bestknn = currmodel

        # tune max depth, begin at 5 and go to 50 by 5
        for c in range(1, nfolds+1):
            # train decision tree and save most accurate model
            currmodel = trainDTC(trainingset, dtclabel, 5*c)
            accuracy = validate(currmodel, validationset, dtclabel)
            if accuracy > dtcmaxacc:
                dtcmaxacc = accuracy
                bestdtc = currmodel
    # Plot corresponding lists for the accuracy vs sample plots

    return bestknn, bestdtc

# get data
data = "./archive/data.csv"
dtclabel = 'mode'
knnlabel ='explicit'
kfold = 10
thresholds = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

# clean data, retrieve 1000 random samples
cleanedData = preprocess(data, 1000)

# train models
knn, dtc = train(cleanedData, kfold, knnlabel, dtclabel)

# get test data, retrieve 1000 random samples
testdata = preprocess(data, 1000)

# print samples vs accuracy
samplevsaccuracy(testdata, kfold, knnlabel, dtclabel)

# BEGIN ROC
# get samples from validation data
dtcsamples = testdata.drop([dtclabel], axis=1)
knnsamples = testdata.drop([knnlabel], axis=1)

# get predictions based on validation samples
dtcpredictions = pd.DataFrame(dtc.predict(dtcsamples), columns=[dtclabel])
knnpredictions = pd.DataFrame(knn.predict(knnsamples), columns=[knnlabel])

# get probabilities
dtcprobabilities = pd.DataFrame(dtc.predict_proba(dtcsamples))
knnprobabilities = pd.DataFrame(knn.predict_proba(knnsamples))

# get tpr and fpr ratios
dtctpr, dtcfpr = ROCplot(dtcpredictions, dtcprobabilities, dtclabel, thresholds)
knntpr, knnfpr = ROCplot(knnpredictions, knnprobabilities, knnlabel, thresholds)

# plot ROC curves
plt.plot(dtcfpr, dtctpr, color='red', label='ROC')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('DTC')
plt.show()

plt.plot(knnfpr, knntpr, color='red', label='ROC')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('KNN')
plt.show()
