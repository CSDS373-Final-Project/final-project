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
    
Outputs:

Questions we want to answer:
#Which model has the smallest MAE?
    #Did feature selection improve the models predictions?
    #How many features did the model use in its predictions that yielded

"""