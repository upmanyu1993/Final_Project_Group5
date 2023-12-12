import streamlit as st
from sample import run_lyrics_generator, read_csv_to_string, calculate_cosine_similarity
from lyrics_analysis import get_summarized_lyrics, get_keywords
from sentence_transformers import SentenceTransformer
import joblib
from tensorflow.keras.models import load_model
import time
import numpy as np
import xgboost as xgb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd

def main():
    st.set_page_config(page_title="Lyric Genie")
    artist = pd.read_csv('artist_feature_means.csv')
    st.header("NLP Final Project : ")
    st.title(" 🧞‍♂️ :rainbow[Lyrics Genie] ")

    # Stylish Display for Team Members
    st.write(" ")
    multi = ''' **Project Team Superstars**:    
      👩‍💻 Jiwoo Suh  
      👩‍💻 Sanjana Godolkar  
      👨‍💻 Upmanyu Singh  
    '''
    st.markdown(multi)
    # st.write(" 👨‍👧‍👧 Project Team Superstars: Jiwoo Suh & Sanjana Godolkar & Upmanyu Singh")
    # st.write("")
    st.divider()
    # st.markdown("<h3 style='text-align: center; color: blue;'>Jiwoo Suh 🚀</h3>", unsafe_allow_html=True)
    # st.markdown("<h3 style='text-align: center; color: blue;'>Sanjana Godolkar 🚀</h3>", unsafe_allow_html=True)
    # st.markdown("<h3 style='text-align: center; color: blue;'>Upmanyu Singh 🚀</h3>", unsafe_allow_html=True)
    # st.header("👷 Let's Build Artist-Specific Lyric Generators")
    df = pd.read_csv('artist10df.csv')

    # Calculate artist lyrics count
    artist_lyrics_count = df.groupby('track_artist')['lyrics'].count().reset_index()
    artist_lyrics_count.columns = ['track_artist', 'lyrics_count']
    artist_lyrics_count = artist_lyrics_count.sort_values(by='lyrics_count', ascending=False)
    artist_lyrics_count10 = artist_lyrics_count[:10]
    artist_lists = artist_lyrics_count10.track_artist.to_list()

    # st.header("Artist Lyrics Analysis")

    # Display the dataframe
    st.header("👩‍🎤 Top 10 Artists by Lyrics Count")
    col1, col2, col3 = st.columns([1,2,1])
    col2.dataframe(artist_lyrics_count10,
                   hide_index=True,
                   use_container_width=True)

    # Section for Lyrics Generation
    st.header("🎵 Generate Your Hit Lyrics ")
    artist_names = ['Queen', 'David Guetta', 'Drake', "Guns N' Roses", 'Logic', 'The Chainsmokers', 'Martin Garrix', '2Pac', 'The Weeknd', 'Eminem']
    selected_artist = st.selectbox("Pick your artist 🎤:", artist_names, index=0, help="Select an artist to mimic their style!")
    starting_lyrics = st.text_area("Kickstart your lyrics 📝:", help="Enter the first few lines of your masterpiece!")

    # Button to generate lyrics
    if st.button('🎡 Spin the Lyrics Wheel '):
        model_path = 'models/results2_'
        if starting_lyrics:
            generated_lyrics = run_lyrics_generator(model_path, [selected_artist], starting_lyrics)
            formatted_lyrics = generated_lyrics.replace("\n", "<br><br>")
            
            container = st.container()
            container.markdown(f"<div style='background-color:#F5F5F5;padding:10px;border-radius:10px;text-align: center;'><h3>💘 Your Lyrics Creation </h3><p>{formatted_lyrics}</p></div>", unsafe_allow_html=True)
            
            st.header("🎵Lyric Similarity Evaluation", divider='rainbow')
            real_lyrics = read_csv_to_string(f"train_{selected_artist.lower().replace(' ', '_')}_dataset.csv")
            st.subheader(f"Artist: {selected_artist}")
            # st.text("🎵Generated Lyrics:")
            # st.markdown(f"<div style='text-align: center;'><b>{generated_lyrics}</b></div>", unsafe_allow_html=True)
            st.text("🎵Comparing Lyrics:")
            st.markdown(f"<div style='text-align: center;'><b>{real_lyrics[:1000]}</b></div>", unsafe_allow_html=True)

            # Calculate and display cosine similarity
            similarity = calculate_cosine_similarity(selected_artist, generated_lyrics)
            # st.header(f"🌟 Cosine Similarity: {similarity}")
            st.success(f"🌟 Cosine Similarity: {similarity}")
            # st.metric(label="Cosine Similarity", value=similarity)
            
            artist = artist[artist['artist']=='David Guetta']
            artist = artist.iloc[:,1:].values.flatten()
            # np.concatenate((embedding, ))
            # Regression Score Logic
            st.header("📊 Lyrics Quality Score ")
            st.write("Calculating the magic of your lyrics...")
            progress = st.progress(0)
            model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            embedding = model.encode(generated_lyrics, normalize_embeddings=True)
            pca = joblib.load('pca_model.joblib')
            
            additional_features = 5613 - 768 -10
            extra_features = np.zeros(additional_features)
            expanded_embedding = np.concatenate((embedding, extra_features, artist))
            transformed_data = pca.transform(expanded_embedding.reshape(1,-1))

            pca_c = joblib.load('pca_model_sanj.joblib')
            expanded_embedding_c = np.concatenate((embedding, np.zeros(8217-768-10), artist))
            transformed_data_c = pca_c.transform(expanded_embedding_c.reshape(1,-1))
            
            model_c = xgb.XGBClassifier()
            model_c.load_model('xgb_model.json')
            category = model_c.predict(transformed_data_c)
            dictionary = {0: 'Low popularity', 1:'Average popularity', 2:'Mid popularity',3:'High popularity'}
            lyrics_model = load_model('lyrics_model.h5')
            regression_score = lyrics_model.predict(transformed_data)
            
            for i in range(101):
                progress.progress(i)
                time.sleep(0.05)
            st.success(f"🌟 Popularity Score: {regression_score[0][0]:.2f}")
            st.success(f"🌟Popularity Category: {dictionary[category[0]]}")

    # Result display
    import json
    file_path = 'result_data.json'
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    result_df = pd.DataFrame(json_data)
    st.header("🤓 Example Results")
    st.dataframe(result_df, hide_index=True)

    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART")
    model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART")
    # Lyrics Analysis Section
    st.header("🔍 Dive Deep into Lyrics Analysis ")
    lyrics = st.text_area("Paste the lyrics for analysis 🧐:", help="Enter lyrics to analyze their essence!")


    if st.button('🔬 Conduct Analysis️'):
        if model and tokenizer and lyrics and summarizer:
            st.subheader('📜 Summarized lyrics')
            # Assume get_summarized_lyrics is defined and works correctly
            summ = get_summarized_lyrics(lyrics, summarizer)
            st.markdown(f"<div style='text-align: center;'><b>{summ}</b></div>", unsafe_allow_html=True)
            
            st.subheader('🔑 Keyword extraction')
            # Assume get_keywords is defined and works correctly
            keyword = get_keywords(lyrics)
            st.markdown(f"<div style='text-align: center;'><b>{keyword}</b></div>", unsafe_allow_html=True)
        else:
            st.error('Please enter lyrics for analysis.')

    

if __name__ == "__main__":
    main()
