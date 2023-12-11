import streamlit as st
from sample import run_lyrics_generator, calculate_cosine_similarity, read_csv_to_string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd
from lyrics_analysis import get_summarized_lyrics, get_summarization, get_keywords



def main():
    st.title("Lyric Genie")
    st.subheader("Building Artist-Specific Lyric Generators", divider = 'rainbow')
    df = pd.read_csv('artist10df.csv')

    # Calculate artist lyrics count
    artist_lyrics_count = df.groupby('track_artist')['lyrics'].count().reset_index()
    artist_lyrics_count.columns = ['track_artist', 'lyrics_count']
    artist_lyrics_count = artist_lyrics_count.sort_values(by='lyrics_count', ascending=False)
    artist_lyrics_count10 = artist_lyrics_count[:10]
    artist_lists = artist_lyrics_count10.track_artist.to_list()

    # st.header("Artist Lyrics Analysis")

    # Display the dataframe
    st.subheader("Top 10 Artists by Lyrics Count")
    col1, col2, col3 = st.columns([1,2,1])
    col2.dataframe(artist_lyrics_count10,
                   hide_index=True,
                   use_container_width=True)
    #
    # # Display the names of the students
    # st.header("Project Team Members:")

    # Input fields for lyrics generation
    st.header("Generate Lyrics")

    # Display the list of artists
    # st.header("Top 10 Artists List")
    # st.write(artist_lists)
    model_path = 'results_'
    # artist_name = st.text_input("Enter artist name:")
    # starting_lyrics = st.text_area("Enter starting lyrics:")

    # Button to generate lyrics
    # if st.button('Generate Lyrics'):
    #     if model_path and artist_name and starting_lyrics:
    #         generated_lyrics = run_lyrics_generator(model_path, [artist_name], starting_lyrics)
    #         # formatted_lyrics = generated_lyrics.replace("\n", "<br>")
    #         # st.markdown(f"<b>{formatted_lyrics}</b>", unsafe_allow_html=True)
    #         formatted_lyrics = generated_lyrics.replace("\n", "<br><br>")
    #         # Center-aligning the lyrics
    #         st.markdown(f"<div style='text-align: center;'><b>{formatted_lyrics}</b></div>", unsafe_allow_html=True)
    #
    #     else:
    #         st.error("Please fill in all the fields.")

    artist_names = ['Queen', 'David Guetta', 'Drake', "Guns N' Roses", 'Logic', 'The Chainsmokers', 'Martin Garrix', '2Pac', 'The Weeknd', 'Eminem']
    selected_artist = st.selectbox("Select an artist:", artist_names)
    starting_lyrics = st.text_area("Enter starting lyrics:")

    # Button to generate lyrics
    if st.button('Generate Lyrics'):
        if starting_lyrics:
            # model_path = f'results_{selected_artist.lower().replace(" ", "_")}'
            generated_lyrics = run_lyrics_generator(model_path, [selected_artist], starting_lyrics)
            formatted_lyrics = generated_lyrics.replace("\n", "<br><br>")
            # Center-aligning the lyrics
            container = st.container(border=True)
            container.markdown(f"<div style='text-align: center;'><b>{formatted_lyrics}</b></div>", unsafe_allow_html=True)

            st.header("Lyric Similarity Evaluation", divider='rainbow')
            real_lyrics = read_csv_to_string(f"{selected_artist.lower().replace(' ', '_')}_df.csv")
            st.subheader(f"Artist: {selected_artist}")
            st.text("Generated Lyrics:")
            st.markdown(f"<div style='text-align: center;'><b>{generated_lyrics}</b></div>", unsafe_allow_html=True)
            st.text("Real Lyrics:")
            st.markdown(f"<div style='text-align: center;'><b>{real_lyrics[:1000]}</b></div>", unsafe_allow_html=True)

            # Calculate and display cosine similarity
            similarity = calculate_cosine_similarity(selected_artist, generated_lyrics)
            st.subheader(f"Cosine Similarity: {similarity}")
        else:
            st.error("Please enter starting lyrics.")

    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART")
    model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART")

    st.header("Lyrics Analysis")
    lyrics = st.text_input("Enter lyrics:")
    
    if st.button('Analysis'):
        if model and tokenizer and lyrics and summarizer:
            st.write('Summerized lyrics')
            summ = get_summarized_lyrics(lyrics, summarizer)
            st.markdown(f"<div style='text-align: center;'><b>{summ}</b></div>", unsafe_allow_html=True)
            
            st.write('Keyword extraction')
            keyword = get_keywords(lyrics)
            st.markdown(f"<div style='text-align: center;'><b>{keyword}</b></div>", unsafe_allow_html=True)
        else:
            st.error('Please fill in all the fields.')


    # Rest of your team members and details
    st.write("2. Sanjana")
    # ... rest of your code ...

if __name__ == "__main__":
    main()
