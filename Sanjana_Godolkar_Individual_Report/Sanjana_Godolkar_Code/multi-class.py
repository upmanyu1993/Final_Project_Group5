import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
from transformers import BertTokenizer, BertModel
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import joblib


# Load and prepare the data
df = pd.read_csv('final_data.csv')
df.drop_duplicates(['track_name', 'track_artist', 'lyrics.1'], inplace=True)

# Select numerical and categorical columns
numerical_cols = ['danceability', 'energy', 'loudness', 'tempo', 'duration_ms', 'key', 'mode', 'acousticness', 'instrumentalness', 'liveness', 'valence']
categorical_cols = ['track_artist', 'playlist_genre', 'track_album_release_date']

# Define transformers
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Fitting the preprocessor
X = df[numerical_cols + categorical_cols]
preprocessor.fit(X)

# Apply transformation
X_transformed = preprocessor.transform(X)

# Convert track popularity to categorical class labels
df['popularity_class'] = df['track_popularity'].apply(lambda x: 0 if x <= 23 else (1 if x <= 47 else (2 if x <= 61 else 3)))

# Function to get BERT embeddings
def get_bert_embeddings(sentences, model_name="prajjwal1/bert-tiny"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.pooler_output[0].numpy())
    
    return np.array(embeddings)

# Load embeddings if already saved
lyrics_embeddings = np.load('lyrics_embeddings.npy')

# Combine embeddings with other preprocessed features
X_transformed_dense = X_transformed.toarray()
combined_features = np.concatenate([lyrics_embeddings, X_transformed_dense], axis=1)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=200)
pca.fit(combined_features)
# Save the PCA model
joblib.dump(pca, 'pca_model_sanj.joblib')

reduced_features = pca.transform(combined_features)

# Extract labels
labels = df['popularity_class'].values

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(reduced_features, labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Fit PCA on training data
# pca = PCA(n_components=0.95)
# pca.fit(X_train)

# Initialize and fit XGBoost classifier with best parameters
xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    learning_rate=0.1,
    max_depth=3,
    n_estimators=200,
    subsample=0.8
)

# Fit the model on the training data
xgb_clf.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", accuracy)

# Save the XGBoost model
xgb_clf.save_model('xgb_model.json')
print("XGBoost Model saved successfully.")


from sklearn.metrics import classification_report

# Generate the classification report
report = classification_report(y_test, y_pred)

# Print the classification report
print(report)
