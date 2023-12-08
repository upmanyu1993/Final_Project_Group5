#Imports
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Loading dataset
data_frame = pd.read_csv('final_data.csv')

#Modifying df
df = data_frame[['track_name', 'track_artist','track_album_name','track_album_release_date','playlist_name','playlist_genre','playlist_subgenre','danceability','energy','key','mode','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','lyrics.1','track_popularity']]
df.drop_duplicates(subset=['track_name'],keep='first',ignore_index=True)
df.rename(columns={'lyrics.1': 'lyrics'}, inplace=True)


################[EDA and PREPROCESSING]######################
num_rows = df.shape[0]
num_cols = df.shape[1]
print(num_rows)
print(num_cols)
print(df.columns)

# Convert 'track_popularity' to Multiclass
df['target'] = df['track_popularity'].apply(lambda x: 0 if x <= 25 else (1 if x <= 50 else (2 if x <= 75 else 3)))

# Drop the original 'track_popularity' column
df = df.drop(columns=['track_popularity'])

# Encode categorical variables using Label Encoding
label_encoder = LabelEncoder()
df['playlist_genre'] = label_encoder.fit_transform(df['playlist_genre'])
df['playlist_subgenre'] = label_encoder.fit_transform(df['playlist_subgenre'])

# Lowercasing
df['lyrics'] = df['lyrics'].str.lower()

# Tokenization
df['lyrics'] = df['lyrics'].apply(word_tokenize)

# Remove stop words
stop_words = set(stopwords.words('english'))
df['lyrics'] = df['lyrics'].apply(lambda x: [word for word in x if word not in stop_words])

# Lemmatization
lemmatizer = WordNetLemmatizer()
df['lyrics'] = df['lyrics'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Join the cleaned text into a single string (if needed)
df['lyrics'] = df['lyrics'].apply(lambda x: ' '.join(x))

# Show the preprocessed data
print(df.head())

#########[SPLIT]############

# Split the data into features and target
X = df.drop(columns=['target'])
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform any additional preprocessing steps, such as scaling numeric features
numeric_columns = X_train.select_dtypes(include=['number']).columns
scaler = StandardScaler()
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

