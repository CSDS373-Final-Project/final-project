import spotipy
from spotipy.oauth2 import SpotifyClientCredential


client_id_list = ['a681eb56d3674a7dbbfc3fc2b61ab419']
client_secret_list = ['28109a4001d143d9bea31b0e331318c1']

client_id = client_id_list[0]
client_secret = 'API--secret--here'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) #spotify object to access API


