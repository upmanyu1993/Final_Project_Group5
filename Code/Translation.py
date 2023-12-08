#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from deep_translator import GoogleTranslator

#%%
# Assuming df is your DataFrame
# Load your dataset here
df = pd.read_csv('spotify_songs.csv')

#%%
# Check for Missing Values
print("Missing Values:")
print(df.isnull().sum())
df.dropna(inplace=True)

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Unique Values in Categorical Columns
print("\nUnique Values in Categorical Columns:")
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].nunique()} unique values")


language_counts = df['language'].value_counts()

# Print the counts for each language
print(language_counts)

#%%
def google_translate(text):
    """Translates a song if it is not in english using google translator"""
    google_translator = GoogleTranslator()
    # If text length is less than 5k chars we directly translate
    # If not we divide in 5000 chars slots and translate them separately, then join
    if len(text) <= 5000:
        lyrics = google_translator.translate(text)
        return lyrics
    else:
        segments = textwrap.wrap(text, 5000)
        translation = []
        for segment in segments:
            translation.append(google_translator.translate(segment))
        return ' '.join(translation)

# %%
# Drop unnecessary columns
columns_to_drop = ['track_id',  'track_album_id', 'playlist_name', 'playlist_id']
df = df.drop(columns=columns_to_drop)

#%%
# Convert 'track_popularity' to binary classes (0 or 1)
df['target'] = df['track_popularity'].apply(lambda x: 0 if x <= 49 else 1)

#%%
# Drop the original 'track_popularity' column
df = df.drop(columns=['track_popularity'])

#%%

# Encode categorical variables using Label Encoding
label_encoder = LabelEncoder()
df['playlist_genre'] = label_encoder.fit_transform(df['playlist_genre'])
df['playlist_subgenre'] = label_encoder.fit_transform(df['playlist_subgenre'])

#%%
# Split the data into features and target
X = df.drop(columns=['target'])
y = df['target']

#%%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# Perform additional preprocessing steps, such as scaling numeric features

numeric_columns = X_train.select_dtypes(include=['number']).columns
scaler = StandardScaler()
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])


# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Convert lyrics to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['lyrics'])
X_test_tfidf = tfidf_vectorizer.transform(X_test['lyrics'])

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report
print(classification_report(y_test, y_pred))

# %%
# Assuming 'df' is your preprocessed DataFrame
df.to_csv('preprocessed_data.csv', index=False)

# %%
