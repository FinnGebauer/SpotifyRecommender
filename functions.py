import pandas as pd
import numpy as np
import spotipy
from scipy.spatial.distance import cosine
from textblob import TextBlob
from spotipy.oauth2 import SpotifyClientCredentials
from spotify_credentials import  SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET

auth_sp = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_sp)


def df_from_recommendations(recs):

  """
  This function returns a dataframe containing track, artist and audio information about songs in a playlist.
  Input: A list containing a dictionary called from the spotify API and already accessed with ["tracks"]

  """


  data_dict = {
    "id":[],  
    "track_name":[], 
    "artist_name":[],
    "artist_pop":[],
    "album":[],
    "album_img" :[],
    "track_pop":[],
    "preview_url" :[],
    "valence":[], 
    "energy":[],
    "danceability":[],
    "loudness" :[],
    "speechiness":[],
    "acousticness":[],
    "instrumentalness":[],
    "liveness":[],
    "mode":[],
    "key":[],
    "tempo":[],
    "duration_ms":[],
    "time_signature":[],
  }


  
  for track in recs:
    
    #Track Id and mood
        data_dict["id"].append(track["id"])
    
    #Track and artist information

        data_dict["track_name"].append(track["name"])
        artist_uri = track["artists"][0]["uri"]
        artist_info = sp.artist(artist_uri)
        #data_dict["artist_name"].append(track_meta["album"]["artists"][0]["name"])

        data_dict["artist_name"].append(track["artists"][0]["name"])
        data_dict["artist_pop"].append(artist_info["popularity"])
        

        #Album
        data_dict["album"].append(track["album"]["name"])
        data_dict["album_img"].append(track["album"]["images"][1]["url"])

        #Popularity of the track
        data_dict["track_pop"].append(track["popularity"])
        if 'preview_url' in track.keys():
          data_dict["preview_url"].append(track["preview_url"])
        else:
          data_dict["preview_url"].append("unknown")
        

        #Audio features
        track_features = sp.audio_features(track["id"])[0]
        data_dict["valence"].append(track_features["valence"])
        data_dict["energy"].append(track_features["energy"])
        data_dict["danceability"].append(track_features["danceability"])
        data_dict["loudness"].append(track_features["loudness"])
        data_dict["speechiness"].append(track_features["speechiness"])
        data_dict["acousticness"].append(track_features["acousticness"])
        data_dict["instrumentalness"].append(track_features["instrumentalness"])
        data_dict["liveness"].append(track_features["liveness"])
        data_dict["mode"].append(track_features["mode"])
        data_dict["key"].append(track_features["key"])
        data_dict["tempo"].append(track_features["tempo"])
        data_dict["duration_ms"].append(track_features["duration_ms"])
        data_dict["time_signature"].append(track_features["time_signature"])


  # Store data in pandas dataframe
  analysis_df = pd.DataFrame(data_dict)

  # Drop duplicates
  analysis_df.drop_duplicates(subset = "id", keep = "first", inplace = True)
  return analysis_df



def text_subjectivity(text):
  return TextBlob(text).sentiment.subjectivity

def text_polarity(text):
  return TextBlob(text).sentiment.polarity

def sentiment_analysis(df, text_col):
  """
  Perform sentiment analysis on text
  ---
  Input:
  df (pandas dataframe): Dataframe of interest
  text_col (str): column of interest
  """
  df['subjectivity'] = df[text_col].apply(text_subjectivity)
  df['polarity'] = df[text_col].apply(text_polarity)
  return df


def split_df(df):
  """
  splits a df into an input_df with only numerical values and
  an output_df with only the values people care about
  """
  input_df = df.drop(['id', 'artist_name', 'track_name', 'album', 'album_img', 'preview_url'], axis=1)
  output_df = df[['id', 'artist_name', 'track_name', 'album', 'album_img', 'preview_url']]
  return input_df, output_df

def cosine_similarity(df, input_vec, predictions):
  """"
  This function returns the row-wise cosine similarity between an input vector (i.e. the mood sliders)
  and the vector represented by the prediction probabilities calculated by the machine learning model
  -----------------------------
  df: The DataFrame to which the similarity column is supposed to be added.
  input_vec: the four-dimensional vector describing the four different moods in values between 0 and 1
  predictions: The four data frame columns as a list showing the prediction probabilities calculated by the model
  """

  df['similarity'] = df[predictions].apply(lambda row: 1 - cosine(row, input_vec), axis=1)
  return df

def get_track_id(df):
  """
  Gets track_ids from an output dataframe if it has an id column.
  """
  track_ids = df['id']
  return track_ids

