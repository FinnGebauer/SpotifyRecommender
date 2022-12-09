import streamlit as st
import pandas as pd
import numpy as np
import random
import spotipy
import sklearn
import plotly.express as px
from skmultilearn.problem_transform import ClassifierChain
from scipy.spatial.distance import cosine
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from functions import df_from_recommendations, sentiment_analysis, text_subjectivity, text_polarity, split_df, cosine_similarity, get_track_id
import pickle

# STYLING

st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1526815456940-2c11653292a2?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80");
             background-repeat: no-repeat;
             background-size: cover;
             background-position: center;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

#AUTHENTIFICATION

scope = "playlist-modify-private user-top-read"
user_id = "zliovv9zwmb29hl2eftgqba3v"
sp = spotipy.Spotify(
        auth_manager=spotipy.SpotifyOAuth(    
          scope=scope, username=user_id))

    #-----------
    ###alternative way of authentification, does not allow playlist creation
    #from spotify_credentials import  SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET
    #auth_sp = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
    #sp = spotipy.Spotify(auth_manager=auth_sp)
    #-----------

#LOAD THE MODEL
#load pickled neural network model
m_cc = pickle.load(open("m_cc.sav", "rb"))

#STREAMLIT 

st.title("Spotify mood recommendations")

   

with st.form("mood-recommender"):
    track_artist = st.selectbox('What should you recommendations be based on?', options=['tracks','artists'])

    with st.expander("Advanced options"):
        tempo = st.slider("Tempo of songs", min_value=0, max_value=250, value=120)
        popularity = st.slider("Deep Cuts or Hits?", min_value=0, max_value=100, value=25)
        time_range = st.select_slider("Define the timeframe of your favourites", options=('short_term', 'medium_term', 'long_term'), value='medium_term')
        offset = st.slider("How far should we dig into your favorites? (offset from top)", min_value=0, max_value=50, value=5)
        create_playlist = st.checkbox("Save recommendations to playlist?")
        create_radarplot = st.checkbox("Would you like to see your playlist as a radar plot?")

    if track_artist == 'artists':
        top_artists = sp.current_user_top_artists(limit=5, offset=offset, time_range=time_range)
        artist_ids = []
        for i in top_artists['items']:
            artist_ids.append(i['id'])
    if track_artist == 'tracks':
        top_tracks = sp.current_user_top_tracks(limit=5, offset=offset, time_range=time_range)
        track_ids= []
        for i in top_tracks['items']:
            track_ids.append(i['id'])
    st.subheader("How do you feel?")


    angry = st.slider("angry", min_value=0.0, max_value=1.0, step=0.05)
    calm = st.slider("calm", min_value=0.0, max_value=1.0, step=0.05)
    feelgood = st.slider("feelgood", min_value=0.0, max_value=1.0, step=0.05)
    sad = st.slider("sad", min_value=0.0, max_value=1.0, step=0.05)
    
    submitted = st.form_submit_button("Get recommendations!")
    
# CREATE OUTPUT
if submitted:
    #transform output into a song list and an array for the moods
    if track_artist == 'artists':
        song_list = sp.recommendations(seed_artists=artist_ids, limit=100, country= "DE", target_tempo=tempo, target_popularity=popularity)['tracks']
    else:
        song_list = sp.recommendations(seed_tracks=track_ids, limit=100, country= "DE", target_tempo=tempo, target_popularity=popularity)['tracks']
        
    mood_array = np.array([angry, calm, feelgood, sad])

    #create the full dataframe
    analysis_df = df_from_recommendations(song_list)
    full_df = sentiment_analysis(analysis_df, "track_name")
    #split the dataframe into numeric and string values
    input_df, output_df = split_df(full_df)

    #run the model and add probabilities to the dataframe
    predictions = m_cc.predict_proba(input_df).toarray()
    output = output_df
    output["angry_prob"] = predictions[:,0]
    output["calm_prob"] = predictions[:,1]
    output["feelgood_prob"] = predictions[:,2]
    output["sad_prob"] = predictions[:,3]

    #check which songs in the song list have the highest similarity and sort the dataframe
    output = cosine_similarity(output, mood_array, ["angry_prob", "calm_prob", "feelgood_prob", "sad_prob"])
    output = output.sort_values(by="similarity", ascending=False, ignore_index=True)

    #output the 25 songs with the highest cosine similarity with album covers and preview players
    for i in range(0,25):
        col1, col2, col3 = st.columns([1,3,1])
        with col1:
            st.image(output['album_img'][i], use_column_width='auto')
        with col2:
            st.write(f"{output['artist_name'][i]} - {output['track_name'][i]}")
        with col3:
            st.audio(output['preview_url'][i])
    
    #if create playlist has been checked, write a playlist to user's account
    if create_playlist:
        playlist_name = f"Cool recommendations"
        playlist = sp.user_playlist_create(user=user_id, public=False, name=playlist_name)
        track_ids = get_track_id(output[:25])    
        sp.user_playlist_add_tracks(user=user_id, playlist_id=playlist['id'], tracks=track_ids)
    
    #create radar plot if it has been checked
    if create_radarplot:
        columns_radar= ['valence', 'energy', 'danceability', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']
        radar_df = input_df[columns_radar]
        fig = px.line_polar(radar_df, r=radar_df.values.mean(axis=0), theta=columns_radar, line_close=True)
        fig.update_traces(fill='toself')
        fig.update_layout(polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )), showlegend=False)
        with st.expander("Show radar plot"):
            st.plotly_chart(fig)
   
   #give the option to have a look at the dataframe
    with st.expander("Show Dataframe"):
        st.dataframe(output.drop(['id', 'album_img', 'preview_url'], axis=1))
        st.dataframe(input_df)

