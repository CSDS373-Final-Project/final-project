"""
Name: maxannplaylistprocessor.py
Author: Madison Kekic
Takes in a playlist urls and outputs information about tracks contained in playlist into csv file
Inputs:
    # None by command line; takes in data from clientdata.py
Outputs:
    #New csv file containing information about 5 playlists- each playlist corresponds to a rating
#Notes
    # Used Medium article to help with this portion of code: https://medium.com/@samlupton/spotipy-get-features-from-your-favourite-songs-in-python-6d71f0172df0
"""

import clientdata
import spotipy
import pandas
from spotipy.oauth2 import SpotifyClientCredentials




def main():
    #Information created and gathered from spotify's developer portal 
    client_id = 'a681eb56d3674a7dbbfc3fc2b61ab419'
    client_secret = '28109a4001d143d9bea31b0e331318c1'
    uri= 'https://docs.google.com/spreadsheets/d/1FOe4uzK8KMnb9ld8Ie0ci-4Hl6krbzvRrfTkLwk8KJ0/edit?gid=0#gid=0'
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) #spotify object to access API
    


    #Loop through possible ratings
    labels = [5,4,3]
    label_index=0

    #Create dataframe to concat data from multiple playlists to 
    combined_data = pandas.DataFrame()
    #List containing playlists created by Maddy- note that each playlist corresponds to a rating (1-5)
    for id in clientdata.MAXANN_PLAYS:
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

            # #Accessing audio features
            # features = sp.audio_features([track_id])











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
                "label": labels[label_index],
                "track_name": track_name,
                "artists": artist['name'],
                "popularity": popularity,
                "duration": duration,
                "explicit": explicit,
                "track_genre": track_genre
             })
        label_index+=1
        print(label_index)
        #Creates pandas dataframe for data
        new_dataframe = pandas.DataFrame(playlist_data)
        #Concatenates data 
        combined_data = pandas.concat([combined_data, new_dataframe])
    #Creates new csv file with resulting dataframe - index is false because we don't want extra column telling us row index
    combined_data.to_csv("maxann_playlist_data.csv", index = False)
if __name__ == '__main__':
    main()
