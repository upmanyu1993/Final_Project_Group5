import streamlit as st
from sample import run_lyrics_generator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# import pandas as pd
from lyrics_analysis import get_summarized_lyrics, get_summarization, get_keywords



def main():
    st.title("Final Project NLP")

    # Display the names of the students
    st.header("Project Team Members:")
    st.write("1. Jiwoo Suh")

    # Input fields for lyrics generation
    st.header("Generate Lyrics")
    model_path = 'results_'
    artist_name = st.text_input("Enter artist name:")
    starting_lyrics = st.text_area("Enter starting lyrics:")

    # Button to generate lyrics
    if st.button('Generate Lyrics'):
        if model_path and artist_name and starting_lyrics:
            generated_lyrics = run_lyrics_generator(model_path, [artist_name], starting_lyrics)
            # formatted_lyrics = generated_lyrics.replace("\n", "<br>")
            # st.markdown(f"<b>{formatted_lyrics}</b>", unsafe_allow_html=True)
            formatted_lyrics = generated_lyrics.replace("\n", "<br><br>")
            # Center-aligning the lyrics
            st.markdown(f"<div style='text-align: center;'><b>{formatted_lyrics}</b></div>", unsafe_allow_html=True)

        else:
            st.error("Please fill in all the fields.")
    
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
