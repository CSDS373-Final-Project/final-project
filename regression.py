"""
Name: regression.py
Authors: Madison Kekic and Maxann Neiger
This program runs several regression models with added functions such as min-max normalization, hyperparameter tuning, feature selection, etc. to determine which models yields the best results.
The results from this program will inform what model we choose to use in playlist_generator.py
Inputs:
    #path: String; File containing playlist data; note that playlist data must have ratings assigned to it
    #training_percentage: Float; percentage of the data that will be used as the training set
    #seed: Int; Random seed
    #min_max: Bool; Contains information on whether we should perform min max normalization
    #model: String; Which model, forest or neuralnet, to use

Outputs:

Questions we want to answer:
#Which model has the smallest MAE?
    #Did feature selection improve the models predictions?
    #How many features did the model use in its predictions that yielded

"""



import random
import csv
import sys
import math
import pandas
import sklearn
import pandas as pd
import os
import torch
from sklearn.model_selection import KFold
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
import sklearn.feature_selection
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

filepath = sys.argv[1]
percent = float(sys.argv[2])
seed = int(sys.argv[3])
minmax = sys.argv[4]
model = sys.argv[5]



#function to check if alpha or space, created from stackoverflow
def isalpha_or_space(self):
    if self == "":
        return False
    for char in self:
        if not (char.isalpha() or char.isspace()):
            return False
    return True

#scales dataset with minmax normalization
def scale_dataset(dataset):
    dataset_new = dataset.copy()
    for i in range(1, len(dataset_new.columns)):
        col = dataset_new.columns[i]
        maximum = dataset_new[col][dataset_new[col].argmax()]
        minimum = dataset_new[col][dataset_new[col].argmin()]
        if(maximum == minimum):
            dataset_new[col] = float(minimum)
        else:
            dataset_new[col] = (dataset_new[col] - minimum) / (maximum - minimum)
    return dataset_new

#splits the data into testing and training sets
def create_data(filename, training_percentage, seed):
    shuffled = filename.sample(frac=1, random_state=seed)
    total_rows = shuffled.shape[0]
    training_rows = int(training_percentage * total_rows)

    # create the training set
    training = shuffled.iloc[:training_rows, :]
    testing = shuffled.iloc[training_rows:, :]

    # split the training attributes and labels
    training_X = training.drop("label", axis=1)
    training_y = training["label"]

    # split the testing attributes and labels
    testing_X = testing.drop("label", axis=1)
    testing_y = testing["label"]

    return training_X, training_y, testing_X, testing_y

# creates a neural network with one hidden layer with a given number of attributes and labels
def create_network(seed, dataset, neurons):
    # the below code initializes the weights in the hidden and output neurons with random values
    # this line sets the random seed in the t   orch library so that our code is reproducable
    torch.manual_seed(seed)    
    
    # this creates a hidden layer with neurons that each have inputs
    hidden_layer = [
        torch.nn.Linear(len(dataset.columns) - 1, neurons),
        torch.nn.Sigmoid(),
    ]

    # this creates an output layer with 10 neurons (one for each label) that each have 128 inputs
    output_layer = [
        torch.nn.Linear(neurons, 1),
    ]
            
    # combine all the layers into a single list
    all_layers = hidden_layer + output_layer

    # turn the layers into a neural network
    network = torch.nn.Sequential(*all_layers)
    
    return network

# converts a training set into smaller train and validation sets
def create_validation(training_X, training_y, valid_percentage):
    # find the split point between training and validation
    training_n = training_X.shape[0]
    valid_rows = int(valid_percentage * training_n)

    # create the validation set
    valid_X = training_X.iloc[:valid_rows]
    valid_y = training_y.iloc[:valid_rows]

    # create the (smaller) training set
    train_X = training_X.iloc[valid_rows:]
    train_y = training_y.iloc[valid_rows:]

    return train_X, train_y, valid_X, valid_y

# trains a neural network with given training data
def train_network(network, training_X, training_y, rate, verbose=False):
    # split the training data into train and validation
    # Note: use 20% of the original training data for validation
    train_X, train_y, valid_X, valid_y = create_validation(training_X, training_y, 0.2)

    # convert our data to PyTorch objects
    train_X = torch.from_numpy(train_X.values).float()
    valid_X = torch.from_numpy(valid_X.values).float()
    train_y = torch.from_numpy(train_y.values).float()
    valid_y = torch.from_numpy(valid_y.values).float()

    # move the data and model to the GPU if possible
    if torch.cuda.is_available():
        device = torch.device('cuda')

        train_X = train_X.to(device)
        train_y = train_y.to(device)
        valid_X = valid_X.to(device)
        valid_y = valid_y.to(device)

        network = network.to(device)

    # create the algorithm that learns the weight for the network (with a learning rate of rate)
    optimizer = torch.optim.Adam(network.parameters(), lr=rate)

    # create the loss function function that tells optimizer how much error it has in its predictions
    # here we use cross entropy since we have a classification task with more than two possible labels
    loss_function = torch.nn.MSELoss()

    # train for 1000 epochs
    num_epochs = 1000
    for epoch in range(num_epochs):
        # make predictions on the training set and validation set
        train_predictions = network(train_X)

        train_predictions = train_predictions.flatten()

        # calculate the error on the training set
        train_loss = loss_function(train_predictions, train_y)
        train_num = train_loss.item()

        # perform backpropagation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()


        valid_predictions = network(valid_X)
        valid_predictions = valid_predictions.flatten()
        valid_loss = loss_function(valid_predictions, valid_y)
        valid_num = valid_loss.item()


    # make predictions for the given X
    output = network(valid_X)
    predictions = output.flatten()

    # calculate the MAE
    MAE = torch.mean(torch.abs(predictions - valid_y)).item()
    return MAE


#function to check if alpha or space, created from stackoverflow
def isalpha_or_space(self):
    if self == "":
        return False
    for char in self:
        if not (char.isalpha() or char.isspace()):
            return False
    return True

#Neural Net code
if(model == "NeuralNet" or model == "neuralnet" or model == "Neuralnet" or model == "N" or model == "n"):
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_device(0)
    
    #read in csv
    dataset = pandas.read_csv(filepath)

    for label, dtype in dataset.dtypes.items():
            #Categorical if not int or float
            if dtype!="int64" and dtype!="float64":
                #Perform one hot encodings for categorical variables
                column = label
                onehots = pandas.get_dummies(dataset[column], prefix=column, drop_first=True, dtype=int)
                dataset = pandas.concat([dataset.drop(column, axis=1), onehots], axis=1)
    if(minmax == True):
        dataset = scale_dataset(dataset)

    training_X, training_y, testing_X, testing_y = create_data(dataset, percent, seed)

    MAEval = 10000000

    for rate in [0.0001, 0.001, 0.01, 0.1]:
        for neurons in [2, 50, 100, 150, 200, 256]:
            network = create_network(seed, dataset, neurons)
            MAE = train_network(network, training_X, training_y, rate)
            if(MAE < MAEval):
                MAEval = MAE
                bestrate = rate
                bestneurons = neurons

    print(MAEval)

if(model == "forest" or model == "Forest" or model == "f" or model == "F"):
    #read in csv
    dataset = pandas.read_csv(filepath)

    for label, dtype in dataset.dtypes.items():
        #Categorical if not int or float
        if dtype!="int64" and dtype!="float64":
            #Perform one hot encodings for categorical variables
            column = label
            onehots = pandas.get_dummies(dataset[column], prefix=column, drop_first=True, dtype=int)
            dataset = pandas.concat([dataset.drop(column, axis=1), onehots], axis=1)


    # #one-hot encoding
    # for column in dataset:
    #     columnSeriesObj = dataset[column]
    #     if(column != "label"):
    #         first_value = dataset[column].values[0]
    #         value = str(first_value)
    #         if(isalpha_or_space(value)):
    #             onehots = pandas.get_dummies(dataset[column], column, drop_first=True, dtype=int)
    #             dataset = pandas.concat([dataset.drop(column, axis=1), onehots], axis=1)

    if(minmax == True):
        dataset = scale_dataset(dataset)

    #create training and testing data
    training_X, training_y, testing_X, testing_y = create_data(dataset, percent, seed)

    #create validation from training
    if(filepath == "maxann_playlist_data.csv"):
        training_X, training_y, valid_X, valid_y = create_validation(training_X, training_y, 0.3336)
    if(filepath == "maddy_playlist_data.csv"):
        training_X, training_y, valid_X, valid_y = create_validation(training_X, training_y, 0.34)

    

    regr = RandomForestRegressor(random_state = seed, n_estimators = 200, criterion = "absolute_error")
    regr.fit(training_X, training_y)
    predictions = regr.predict(testing_X)
    MAE = mean_absolute_error(valid_y, predictions)
    print(MAE)

    def log_tree(tree, dataset, dataset_filename, train_percentage, seed):
        # create the filename of the new image
        filename = ("tree"
                    + "_" + dataset_infile[:-4]
                    + "_1t"
                    + "_" + str(int(train_percentage * 100)) + "p"
                    + "_" + str(seed) + ".png")

        # get the names of the attribute    s
        attributes = list(dataset.drop("label", axis=1))

        # get the values of the labels
        labels = sorted(list(dataset["label"].unique()))

        # create the image
        fig = plt.figure(figsize=(100, 100))
        plotted = sklearn.tree.plot_tree(tree,
                                        feature_names=attributes,
                                        class_names=labels,
                                        filled=True,
                                        rounded=True)

        # save the image to file
        fig.savefig(filename)




