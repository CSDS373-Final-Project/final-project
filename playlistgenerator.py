"""
Name: playlistgenerator.py
Authors: Madison Kekic and Maxann Neiger
Implements the best performing model from regression.py to generate a playlist based on the users preferences
Command Line Inputs:
    #path: String- A path to file containing data based on user preferences (given datasets: maxann_playlist_data.csv and maddy_playlist_data.csv)
    #song_num: Int- The number of songs you'd like to generate in your playlist
    #include_rate: Bool- Include prediction of how much the algorithm predicts you will like a given song
If no command line args are given, the user will be prompted with a menu
Outputs:
    #Playlist outputted to terminal
"""

import sys
import sklearn.ensemble
import spotipy
import pandas
from spotipy.oauth2 import SpotifyClientCredentials
from pandas.api.types import is_numeric_dtype


   
client_id = 'a681eb56d3674a7dbbfc3fc2b61ab419'
client_secret = '28109a4001d143d9bea31b0e331318c1'
uri= 'https://docs.google.com/spreadsheets/d/1FOe4uzK8KMnb9ld8Ie0ci-4Hl6krbzvRrfTkLwk8KJ0/edit?gid=0#gid=0'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) #spotify object to access API

"""
Processes Command Line Arguments
Inputs: None
Outputs:
"""

def process_input():
    if len(sys.argv) == 1:
        path, song_num, include_rate = menu()
        return path, song_num, include_rate
    elif len(sys.argv)!=4 and len(sys.argv)!=1:
        print("Incorrect arguments provided. Expected 1 or 4 arguments, but got", len(sys.argv), "Please pass in exactly 4 command line arguments including playlistgenerator.py, path, number of songs, and whether or not to include rating. If you want to be prompted with a menu, please run with no additional arguments")
        quit()
    else:
        path = sys.argv[1]
        song_num = sys.argv[2]
        include_rate = sys.argv[3] 
        print(path)
 
        try:
            #Checks that percentage and seeds are integers, if not throws a value error
            song_num = int(song_num)
            #Standardizes min_max string and assures that string corresponds to true or false
            include_rate = include_rate.lower()
            if include_rate.startswith("t"):
                include_rate = True
                return path, song_num, include_rate
            elif include_rate.startswith("f"):
                include_rate = False
                return path, song_num, include_rate
            else:
                print("Unexpected value for include_rate variable; expected 'true' or 'false', but got", include_rate, " Please try again.")
                exit()
            
        except ValueError:
            print("Value error: expected integer value for song_num, but got ", song_num) 
            exit()


"""
Prints a menu if a user wishes to use their own music
"""
def menu():
    print("Welcome to playlistgenerator.py! Please follow the next prompts to generate.")
    done = False
    playlists=[]
    ratings = []
    while not done: 
        print("Please note that this algorithm works best if you have at least 3 playlists with each playlist corresponding to a rating 1-5")
        url = input("Please enter a spotifyplaylist url in the form: open.spotify.com/playlist/7dKIzrS0qkRFFveqvXBRuX. Hit enter to stop inputting playlists.")       
        #Signifies user wants to quit menu
        if url=="":
            #Requires user to input at least one playlist
            if len(playlists)==0:
                print("Please input at least one playlist")
            else:
                done = True
        else:
            parts = url.split("/")
            #Playlist id should be 22 characters long
            if len(parts[2])==22:
                playlists.append(parts[2])
            else:
                print("Playlist id is invalid.")
    
    play_index = 0
    while len(ratings)!=len(playlists):
        results = sp.playlist(playlists[play_index], fields="name")
        play_name = results["name"]
        try: 
            rating = int(input(f"Assign a rating to the songs in the playlist '{play_name}': "))

            if 1<=rating<=5:
                ratings.append(rating)
                play_index+=1
            else:
                print("Invalid rating; expected rating from 1-5, but got", rating)
                continue
        except ValueError:
            print("Expected integer value for rating, but got ", rating)
            continue
    
    path_name = create_csv(playlists,ratings)

    valid = False
    while not valid:
        try:
            song_num = int(input("How many songs should be in the new playlist?"))
            valid = True
        except ValueError:
            print("Expected int value for song_num, but got ", song_num)
        
    boolean = False
    while not boolean:
        include_rate = input("Should the model include the model's predicted rating?")

        include_rate = include_rate.lower()
        if include_rate.startswith("t"):
            include_rate = True
        elif include_rate.startswith("f"):
            include_rate = False

        else:
            print("Unexpected value for include_rate variable; expected 'true' or 'false', but got", include_rate, " Please try again.")
    return path_name, song_num, include_rate
        

"""

"""
def create_csv(playlists,ratings):
    combined_data = pandas.DataFrame()
    rate_ind = 0
    #List containing playlists created by Maddy- note that each playlist corresponds to a rating (1-5)
    for id in playlists:
        #Get playlist id from url


        #Fetch playlist and create new list to append data about tracks in playlist
        results = sp.playlist_tracks(id)

        playlist_data = []

        #Looping over songs in playlist and extracting information     
        for item in results['items']:
            track = item['track']
            track_id = track['id']
            track_name = track['name']


            #Takes only the first artists name
            artist = track['artists'][0]

            #Extracts basic data about track
            popularity = track['popularity']
            duration = track['duration_ms']
            explicit = track['explicit']    


            #Accessing artist information to determine track genre
            artist_id = artist['id']
            artist_info = sp.artist(artist_id)
            genres = artist_info.get('genres', [])
            if len(genres)>=1:
                track_genre=genres[0]
            else:
                track_genre = None

         
            playlist_data.append({
                #Reminder that this starts at index 0, which is 5
                "label": ratings[rate_ind],
                "track_name": track_name,
                "artist": artist['name'],
                "popularity": popularity,
                "duration": duration,
                "explicit": explicit,
                "track_genre": track_genre
             })
        rate_ind+=1
        #Creates pandas dataframe for data
        new_dataframe = pandas.DataFrame(playlist_data)
        #Concatenates data 
        combined_data = pandas.concat([combined_data, new_dataframe])
    #Creates new csv file with resulting dataframe - index is false because we don't want extra column telling us row index
    combined_data.to_csv("new_playlist_data.csv", index = False)
    return "new_playlist_data.csv"

"""
Chooses 10k songs from test file
"""
def process_test_file():
    #Used Copilot GPT 5 to find out how to get information stored at index i printed to the terminal
    test_data = pandas.read_csv("test11k.csv")
    shuffled = test_data.sample(frac=1)
    selected_raw = shuffled.iloc[:100].copy()
    # keep metadata for display
    song_metadata = selected_raw[["track_name", "artists", "track_genre"]].copy()
    test_data = one_hots(selected_raw)
    # test_data = scale_dataset(test_data)
    return test_data, song_metadata
    
# """
# Performs min max normalization
# """
# def scale_dataset(dataset):
#     dataset_new = dataset.copy()
#     # skip first column (label) and non-numeric columns
#     for col in dataset_new.columns[1:]:
#         if not is_numeric_dtype(dataset_new[col]):
#             continue
#         maximum = dataset_new[col].max()
#         minimum = dataset_new[col].min()
#         if maximum == minimum:
#             dataset_new[col] = float(minimum)
#         else:
#             dataset_new[col] = (dataset_new[col] - minimum) / (maximum - minimum)
#     return dataset_new
"""
"""
def split_data(train_data, test_data):
    # split the training attributes and labels
    train_x= train_data.drop("label", axis=1)
    train_y = train_data["label"]

    # split the testing attributes and labels
    test_x= test_data.drop("label", axis=1)
    test_y = test_data["label"]

    return train_x, train_y, test_x, test_y

# def create_validation(train_x, train_y, percent):
#     # find the split point between training and validation
#     training_n = train_x.shape[0]
#     valid_rows = int(percent * training_n)

#     # create the validation set
#     valid_X = train_x.iloc[:valid_rows]
#     valid_y = train_y.iloc[:valid_rows]

#     # create the (smaller) training set
#     train_X = train_x.iloc[valid_rows:]
#     train_y = train_y.iloc[valid_rows:]

#     return train_X, train_y, valid_X, valid_y

def one_hots(dataset):
    for label, dtype in dataset.dtypes.items():
        #Categorical if not int or float
        if dtype!="int64" and dtype!="float64":
            #Perform one hot encodings for categorical variables
            column = label
            onehots = pandas.get_dummies(dataset[column], prefix=column, drop_first=True, dtype=int)
            dataset = pandas.concat([dataset.drop(column, axis=1), onehots], axis=1)
    return dataset


#3 depth #tree_num 50
def run_model(training_x, training_y, testing_x, song_meta):
    testing_x = testing_x.reindex(columns=training_x.columns, fill_value=0)
    regr = sklearn.ensemble.RandomForestRegressor(n_estimators = 50, criterion = "absolute_error", max_depth = 3)
    regr.fit(training_x, training_y)
    predictions = regr.predict(testing_x)

    results={}
    prediction_index = 0
    print(testing_x)
    for i, pred in enumerate(predictions):  
        row = song_meta.iloc[i]
        song_display = f"{row.track_name} — {row.artists}"
        if pred in results:
            results[pred].append(song_display)
        else:
            results[pred] = [song_display]

        prediction_index+=1
    return results

def decide_songs(results, song_num):
    pass


def print_model():
    pass

def main():
    test_data, song_metadata = process_test_file()
    path, song_num, include_rate = process_input()
    train_data = pandas.read_csv(path)
    train_data = one_hots(train_data)
    train_data = train_data.sample(frac = 1)
    train_x, train_y, test_x, test_y = split_data(train_data, test_data)

    results = run_model(train_x, train_y, test_x, song_metadata)
    print(results)




if __name__ == '__main__':
    main()