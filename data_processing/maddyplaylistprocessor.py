"""
Name: maddyplaylistprocessor.py
Author: Madison Kekic
Takes in a playlist urls and outputs information about tracks contained in playlist into csv file
Inputs:
    # None by command line; takes in data from clientdata.py
Outputs:
    #New csv file containing information about 5 playlists- each playlist corresponds to a rating
#Notes
    # Used Medium article to help with this portion of code: https://medium.com/@samlupton/spotipy-get-features-from-your-favourite-songs-in-python-6d71f0172df0
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredential
import clientdata
import pandas


def main():
    #Information created and gathered from spotify's developer portal 
    client_id = clientdata.MADDY_ID
    client_secret = clientdata.MADDY_SECRET
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) #spotify object to access API



    #Loop through possible ratings
    labels = [5,4,3,2,1]
    label_index=0

    #Create dataframe to concat data from multiple playlists to 
    combined_data = pandas.DataFrame()
    #List containing playlists created by Maddy- note that each playlist corresponds to a rating (1-5)
    for url in clientdata.MADDY_URLS:
        #Get playlist id from url
        playlist_id = url.split("/")[-1].split("?")[0]

        #Fetch playlist and create new list to append data about tracks in playlist
        results = sp.playlist_tracks(playlist_id)
        playlist_data = []

        #Looping over songs in playlist and extracting information     
        for item in results['items']:
            track = item['track']
            track_id = track['id']
            track_name = track['name']

            #Takes only the first artists name
            artist = track['artists'][0]['name']
            
            #Extracts basic data about track
            popularity = track['popularity']
            duration = track['duration_ms']
            explicit = track['explicit']    

            #Accessing audio features
            features = sp.audio_features(track_id)[0]

            #Accessing artist information to determine track genre
            artist_id = artist['id']
            artist_info = sp.artist(artist_id)
            genres = artist_info.get('genres', [])

            if genres:
                track_genre=genres[0]
            else:
                track_genres = None

            #Appending data to playlist_data
            if features:
                playlist_data.append({
                    #Reminder that this starts at index 0, which is 5
                    "label": labels[label_index],
                    "track_name": track_name,
                    "artist": artist,
                    "popularity": popularity,
                    "duration": duration,
                    "explicit": explicit,
                    "danceability": features['danceability'],
                    "energy": features['energy'],
                    "loudness": features['loudness'],
                    "speechiness": features['speechiness'],
                    "acousticness": features['acousticness'],
                    "tempo": features['tempo'],
                    "track_genre": track_genre
                })
            label_index+=1
            #Creates pandas dataframe for data
            new_dataframe = pandas.DataFrame(playlist_data)
            #Concatenates data 
            combined_data = pandas.concat([combined_data, new_dataframe])
    #Creates new csv file with resulting dataframe - index is false because we don't want extra column telling us row index
    combined_data.to_csv("maddy_playlist_data.csv", index = False)
if __name__ == '__main__':
    main()