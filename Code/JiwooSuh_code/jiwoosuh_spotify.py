import re
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
from transformers import pipeline
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

cwd = os.getcwd()
os.chdir('/home/ubuntu/NLP/Final_Project')

# Read the dataset
data = pd.read_csv('spotify_songs 2.csv')
print(data.info())
print(data.head())

df = data[['track_id', 'track_name', 'track_artist', 'lyrics', 'language', 'track_popularity']]
df = df[df['language'] == 'en']
df.drop(['language'], axis=1, inplace=True)
df.dropna(subset=['lyrics'], inplace=True)
print(data.info())

#%%
# Get Sentiment score using pre-trained model from Hugging Face
# Pre-trained model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Function to get sentiment from lyrics
def get_sentiment(lyrics):
    inputs = tokenizer(lyrics, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    sentiment = model.config.id2label[predicted_class_id]
    return sentiment

# Apply sentiment analysis to the DataFrame
df['sentiment'] = df['lyrics'].apply(get_sentiment)

# Display the DataFrame with the new 'sentiment' column
print(df.head())

#%%
# import torch
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from torch.utils.data import TensorDataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# lyrics = df['lyrics'].values
# popularity = df['track_popularity'].values
# X_train, X_test, y_train, y_test = train_test_split(lyrics, popularity, test_size=0.2)
# X_train = X_train.tolist()
# X_test = X_test.tolist()
#
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
# #%%
# from tqdm import tqdm
#
# train_encodings = []
# for lyric in tqdm(X_train):
#   encoding = tokenizer(lyric, truncation=True, padding=True)
#   train_encodings.append(encoding)
#
# test_encodings = []
# for lyric in tqdm(X_test):
#   encoding = tokenizer(lyric, truncation=True, padding=True)
#   test_encodings.append(encoding)
#
# # train_encodings = tokenizer(X_train, truncation=True, padding=True)
# # test_encodings = tokenizer(X_test, truncation=True, padding=True)
# #%%
# train_labels = torch.tensor(y_train)
# test_labels = torch.tensor(y_test)
#
# train_encodings_tensor = torch.cat([encoding for encoding in train_encodings])
# test_encodings_tensor = torch.cat([encoding for encoding in test_encodings])
# # Create tf training dataset
# train_dataset = TensorDataset(train_encodings_tensor, train_labels)
#
# # Fine-tune model
# trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
# trainer.train()
#
# # Evaluate model on test set
# final_predictions = []
# for input_ids in test_encodings:
#   with torch.no_grad():
#      output = model(input_ids)
#   final_predictions.append(output.logits.argmax().item())
#
# print(accuracy_score(final_predictions, y_test))


#%%
# lyrics summarization
from transformers import pipeline

summarizer = pipeline("summarization", model="Falconsai/text_summarization")

# Function to get summarized lyrics
def get_summarized_lyrics(lyrics):
    summary = summarizer(lyrics, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# # Apply summarization to the DataFrame
# df['summarized_lyrics'] = df['lyrics'].apply(get_summarized_lyrics)
#
# # Display the DataFrame with the new 'sentiment' and 'summarized_lyrics' columns
# print(df.head())
#%%
# get similar lyrics
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode(sentences)
print(embeddings)

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

#%%
def get_similar_lyrics(lyrics):
    encoded_input = tokenizer(lyrics, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    lyrics_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    lyrics_embeddings = F.normalize(lyrics_embeddings, p=2, dim=1)
    top5result = lyrics_embeddings
    return lyrics_embeddings

#%%
# cosine similarity
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['lyrics'])
print('TF-IDF shape :',tfidf_matrix.shape)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print('Cosine Similarity :',cosine_sim.shape)

title_to_index = dict(zip(df['track_name'], df.index))


def get_recommendations(trackname, cosine_sim=cosine_sim):
    idx = title_to_index[trackname]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    track_indices = [idx[0] for idx in sim_scores]
    return df['track_name'].iloc[track_indices]

print(get_recommendations('I Feel Alive'))

#%%
# Langchain - train lyrics by artist and generate lyrics
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="Photolens/llama-2-7b-langchain-chat")
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Photolens/llama-2-7b-langchain-chat")
model = AutoModelForCausalLM.from_pretrained("Photolens/llama-2-7b-langchain-chat")

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Group lyrics by artist
artist_lyrics = df.groupby('track_artist')['lyrics'].apply(lambda x: ' '.join(x)).reset_index()

# Pretrained language model
tokenizer = AutoTokenizer.from_pretrained("Photolens/llama-2-7b-langchain-chat")
model = AutoModelForCausalLM.from_pretrained("Photolens/llama-2-7b-langchain-chat")

lyrics_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_lyrics(artist, keyword):
    artist_lyrics_text = artist_lyrics[artist_lyrics['track_artist'] == artist]['lyrics'].values[0]
    prompt = f"{artist} {keyword} lyrics: {artist_lyrics_text}"
    generated_lyrics = lyrics_generator(prompt, max_length=200, temperature=0.8, num_return_sequences=1)[0]['generated_text']
    return generated_lyrics

artist_name = 'Queen'  # Replace with the desired artist from your dataset
user_keyword = 'love'  # Replace with the user-provided keyword
generated_lyrics = generate_lyrics(artist_name, user_keyword)

print(generated_lyrics)
