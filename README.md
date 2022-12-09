# SpotifyRecommender
Project for a Spotify recommender system using multiclass/multilabel classification and cosine similarity

This project was the final project for the data science bootcamp at Spiced academy.

## Data
Using the Spotify API, I pulled playlist matching certain moods. Accodring to Thayer's mood model, I chose angry, happy, calm and sad. I pulled 10 playlists for every mood, totalling 3000 songs.

## Models
I classified the songs using several multiclass models, the best one ending up a random forest model (75% accuracy). Using the predict probabilities of this model, I decided to turn the data set into a multilabel dataset and run a ClassifierChain with a random forest (86% accuracy). 

## Streamlit
The web app runs on streamlit. It takes the current logged in user and gives the option to get recommendations based on 5 top artists or tracks. As advanced options, the user has the possibility to choose among others the tempo of songs, popularity, or has the option to save a playlist to the account. The core feature are the mood sliders which let the user define their emotional state.

The output is a list of 25 songs with album covers and preview players as well as an optional radar plot. The recommendations are made using cosine similarity between the input vector (the mood sliders) and the predict probabilities of the model.
