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

"""
Processes Command Line Arguments
Inputs:
Outputs:
"""

def process_input():
    #If the user does not input any command line arguments, the 
    if len(sys.argv == 1):
        path, song_num, include_rate = menu()
    elif 1<len(sys.argv)<4 or len(sys.argv)>4:
        print("Incorrect arguments provided. Expected 1 or 4 arguments, but got", len(sys.argv), "Please pass in exactly 4 command line arguments including playlistgenerator.py, path, number of songs, and whether or not to include rating. If you want to be prompted with a menu, please run with no additional arguments")
    
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
               
    num_songs = int(input("How many songs would you like to be in your new playlist?"))               
    create_csv(playlists,ratings)


"""
s
"""
def create_csv():
    pass

def run_model():
    pass

def print_model():
    pass

def main():
   
if __name__ == '__main__':
    main()