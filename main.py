# This file contains code required for the preprocessing of data and
# the k-Nearest Neighbours algorithm where k is hyperparameter that will be tuned
# Authors: Jatin Jain

import pandas as pd
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


def kNN(df):
    """Code for applying k-Nearest Neighbours algorithm"""
    y = df[['explicit']]  # Treat explicit column as labels for the classifier
    X = df.drop(['explicit'])  # Drop the column from the main data


data = "./archive/data.csv"
cleanedData = preprocess(data)
kNN(cleanedData)
