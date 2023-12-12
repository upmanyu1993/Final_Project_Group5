from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd

#%%
data = pd.read_csv('spotify_songs 2.csv')
# data = pd.read_csv('final_data.csv')
print(data.info())
print(data.head())
df = data[['track_id', 'track_name', 'track_artist', 'lyrics', 'language', 'track_popularity']]
df = df[df['language'] == 'en']
df.drop(['language'], axis=1, inplace=True)
df.dropna(subset=['lyrics'], inplace=True)
print(df.info())

summarizer = pipeline("summarization", model="Falconsai/text_summarization")

# Function to get summarized lyrics
def get_summarized_lyrics(lyrics):
    summary = summarizer(lyrics, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

#%%
tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART")
model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART")
def get_summarization(lyrics):
    inputs = tokenizer(lyrics, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=30, num_beams=4, length_penalty=2.0, early_stopping=True)
    summarization = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    return summarization

#%%
from keybert import KeyBERT
def get_keywords(lyrics, keyphrase_ngram_range=(1, 1), stop_words='english'):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(lyrics, keyphrase_ngram_range=keyphrase_ngram_range, stop_words=stop_words)
    return keywords

#%%
# Usage
print(get_summarized_lyrics(df['lyrics'].iloc[2]))
print('------------------------')
print(get_summarization(df['lyrics'].iloc[2]))
print('------------------------')
print(get_keywords(df['lyrics'].iloc[2]))

# # Apply summarization to the DataFrame
# df['summarized_lyrics'] = df['lyrics'].apply(get_summarized_lyrics)
#
# # Display the DataFrame with the new 'sentiment' and 'summarized_lyrics' columns
# print(df.head())