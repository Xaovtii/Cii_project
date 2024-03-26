import tensorflow as tf
import streamlit as st
import pandas as pd

# cached resources for model
@st.cache_resource
def load_resources(file_path):
    return tf.saved_model.load(file_path)

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load the song and data CSV files and TensorFlow model
song_df = load_data('data_and_model/songs.csv')
df = load_data('data_and_model/data.csv')
loaded_model = load_resources('data_and_model/model')

# Convert 'city' column to string data type in both DataFrames
df["city"] = df["city"].astype(str)
song_df["name"] = song_df["name"].astype(str)

# Create a list of unique msno_ids
msno_ids = df['msno'].unique().tolist()

# Create a dictionary to map index to msno_id
msno_index_map = {index: msno_id for index, msno_id in enumerate(msno_ids)}

# Display a dropdown to select msno_id
selected_index = st.selectbox("Select msno_id", list(msno_index_map.keys()))

# Get the selected msno_id
selected_msno_id = msno_index_map[selected_index]

# Filter data for the selected msno_id
filtered_data = df[df['msno'] == selected_msno_id]

# # Get the songs the user used to listen to
user_songs = filtered_data['song_id'].unique().tolist()

# get user past listened songs (no predictions)
def get_user_listened_songs(user_data):
    # Filter matching songs from song_df
    matching_songs = song_df[song_df['song_id'].isin(user_songs)]

    # display 
    st.write(f"User with index {selected_index} used to listen to the songs:")
    matching_songs_renamed = matching_songs[['name', 'artist_name','song_length', 'language', 'genre_ids']].rename(
        columns={'name': 'Song Name', 'artist_name': 'Artist','song_length': 'Song Length', 'language': 'Language', 'genre_ids': 'Genre IDs'}
    )

    st.dataframe(matching_songs_renamed)
    
    # unique songs map/dict (name and song_id)
    unique_songs_map = {row['name']: row['song_id'] for index, row in matching_songs.iterrows()}

    return matching_songs_renamed, unique_songs_map

displayed_songs, unique_songs_map = get_user_listened_songs(user_data=filtered_data)

# Function to describe user listening preferences
def describe_user_preferences(user_data):
    # Top artists
    top_artists = user_data['artist_name'].value_counts().nlargest(3).index.tolist()

    # Average song length
    average_song_length = user_data['song_length'].mean()

    # Top genres
    top_genres = user_data['genre_ids'].value_counts().nlargest(3).index.tolist()

    # Top languages
    top_languages = user_data['language'].value_counts().nlargest(3).index.tolist()

    # Display user listening preferences
    st.write("User Listening Preferences:")
    st.write("Top Artists:")
    for i, artist in enumerate(top_artists, 1):
        st.write(f"{i}. {artist}")

    st.write("Average Song Length:")
    st.write(average_song_length)

    st.write("Top Genres:")
    for i, genre in enumerate(top_genres, 1):
        st.write(f"{i}. {genre}")

    st.write("Top Languages:")
    for i, language in enumerate(top_languages, 1):
        st.write(f"{i}. {language}")

# Call the describe_user_preferences function
describe_user_preferences(filtered_data)

# if displayed songs then we can also have select for songs
if displayed_songs is not None:
    selected_song_id = st.selectbox("Select song", list(unique_songs_map.keys()))
    selected_song_id = unique_songs_map[selected_song_id]

    filtered_data = filtered_data[filtered_data['song_id'] == selected_song_id].reset_index(drop=True)
    user_data = {
        'msno': filtered_data['msno'].tolist(),
        'song_id': filtered_data['song_id'].tolist(),
        'source_system_tab': filtered_data['source_system_tab'].tolist(),
        'source_type': filtered_data['source_type'].tolist(),
        'source_screen_name': filtered_data['source_screen_name'].tolist(),
        'city': filtered_data['city'].tolist(),
        'target': filtered_data['target'].tolist(),
        'registered_via': filtered_data['registered_via'].tolist(),
        'gender': filtered_data['gender'].tolist(),
    }

# Generate recommendations function
def generate_recommendations(user_data):
    user_data = {
        'msno': tf.constant(user_data['msno'], dtype=tf.string),
        'song_id': tf.constant(user_data['song_id'], dtype=tf.string),  # Use the selected song ID
        'source_system_tab': tf.constant(user_data['source_system_tab'], dtype=tf.string),
        'source_type': tf.constant(user_data['source_type'], dtype=tf.string),
        'source_screen_name': tf.constant(user_data['source_screen_name'], dtype=tf.string),
        'city': tf.constant(user_data['city'], dtype=tf.string),
        'target': tf.constant(user_data['target'], dtype=tf.int32),
        'registered_via': tf.constant(user_data['registered_via'], dtype=tf.int32),
        'gender': tf.constant(user_data['gender'], dtype=tf.string)
    }

    # Generate recommendations
    scores, titles = loaded_model(user_data)

    if titles.shape[0] == 0:
        st.write("No more recommendations found for the selected song.")
        return

    recommended_song_ids = titles[0].numpy().astype(str)

    # Filter matching songs from song_df
    matching_songs = song_df[song_df['song_id'].isin(recommended_song_ids)]
    # Display information about the selected song
    # Filter matching songs from song_df
    matching_song = song_df[song_df['song_id'] == selected_song_id]
    st.write("Information of Selected Song :")
    selected_song_info = matching_song[['name', 'artist_name', 'genre_ids', 'song_length', 'language']].rename(
        columns={'name': 'Song Name', 'artist_name': 'Artist', 'genre_ids': 'Genre IDs', 'song_length': 'Song Length', 'language': 'Language'}
    )
    st.dataframe(selected_song_info)
    # Filter user data for the selected song
    filtered_user_data = filtered_data[filtered_data['song_id'] == selected_song_id]
    
    st.write(f"More Information of user with index \"{selected_index}\" for the Selected Song:")
    user_info = filtered_user_data[['msno', 'gender', 'source_system_tab', 'source_type', 'source_screen_name', 'city', 'target', 'registered_via']].rename(
        columns={'msno': 'User ID', 'gender': 'Gender', 'source_system_tab': 'Source System Tab', 'source_type': 'Source Type', 'source_screen_name': 'Source Screen Name', 'city': 'City', 'target': 'Target', 'registered_via': 'Registered Via'}
    )
    st.dataframe(user_info)

    selected_song_name = next((name for name, song_id in unique_songs_map.items() if song_id == selected_song_id), None)

    # Display recommendations using Streamlit dataframe
    st.write(f"More recommended songs for the song:  {selected_song_name}:")
    matching_songs_renamed = matching_songs[['name', 'artist_name', 'language', 'genre_ids']].rename(
        columns={'name': 'Song Name', 'artist_name': 'Artist', 'language': 'Language', 'genre_ids': 'Genre IDs'}
    )

    st.dataframe(matching_songs_renamed)

if st.button("Generate Recommendations"):
    if selected_song_id is not None:
        generate_recommendations(user_data=user_data)
    else:
        st.write("Please select a song to generate recommendations.")



    

    