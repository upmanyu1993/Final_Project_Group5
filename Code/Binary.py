# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
from transformers import DistilBertTokenizer, DistilBertModel


# Load dataset
data_frame = pd.read_csv('final_data.csv')

# Modify df
df = data_frame[['track_name', 'track_artist', 'track_album_name', 'track_album_release_date', 
                 'playlist_name', 'playlist_genre', 'playlist_subgenre', 'danceability', 'energy', 
                 'key', 'mode', 'acousticness', 'instrumentalness', 'liveness', 'valence', 
                 'tempo', 'duration_ms', 'lyrics.1', 'track_popularity']]
df.drop_duplicates(subset=['track_name'], keep='first', ignore_index=True)
df.rename(columns={'lyrics.1': 'lyrics'}, inplace=True)

# Convert 'track_popularity' to binary classes (0 or 1) as target
df['target'] = df['track_popularity'].apply(lambda x: 0 if x <= 49 else 1)

# Drop the original 'track_popularity' column
df = df.drop(columns=['track_popularity'])

# Function to create BERT embeddings
def create_distilbert_embeddings(texts):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Preprocess lyrics for BERT
df['lyrics'] = df['lyrics'].apply(lambda x: ' '.join(x))

# Split the data into features and target
X = df.drop(columns=['target'])
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create embeddings for training and testing data
train_embeddings = create_distilbert_embeddings(X_train['lyrics'].tolist())
test_embeddings = create_distilbert_embeddings(X_test['lyrics'].tolist())

# Example: Logistic Regression for binary classification
log_reg = LogisticRegression()
log_reg.fit(train_embeddings, y_train)
log_reg_predictions = log_reg.predict(test_embeddings)
print('Logistic Regression Accuracy:', accuracy_score(y_test, log_reg_predictions))

# Example: Random Forest for binary classification
random_forest = RandomForestClassifier()
random_forest.fit(train_embeddings, y_train)
rf_predictions = random_forest.predict(test_embeddings)
print('Random Forest Accuracy:', accuracy_score(y_test, rf_predictions))
