"""
Name: datapreprocess.py
Authors: Madison Kekic and Maxann Neiger
Inputs:
    # path: File containing data to transform
Outputs:
    # new csv file in file system with the following modifications made:
        # danceability, energy, acousticness, and speechiness converted to be integers and on a 0-100 scale 
        # explicit category converted to dummy variable
        # tempo and loudness converted to integers
        # convert duration in miliseconds to seconds
"""
import sys
import pandas


def create_csv(path):
    ds = pandas.read_csv(path)
    
def main():
    if len(sys.argv) != 2:
        print("Error processing arguments, expected 2 arguments, but got ", len(sys.argv), " arguments instead.")
        exit()
    else:
        path = sys.argv[1]
        create_csv(path)

main()