#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import nltk
import langid
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# # Download NLTK resources if not already downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

#%%
#####LOADING DATASET#########
# Assuming df is your DataFrame
# Load your dataset here
df = pd.read_csv('spotify_songs.csv')

#%%
###########EDA###########
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

# Dropping all songs that aren't english(en)
df = df[df['language'] == 'en']

# Print the counts for each language after dropping
print(language_counts)

# %%
#Upmanyu's preprocessing
# def preprocessing(text):
#     """
#     It takes the string and preprocess the string with the help pf nltk library
#     Parameters:
#         text(str): string which needs to be prerprocessed
#     Return
#         preprocessed string with no stopwords
#     """
#     # from nltk.corpus import stopwords
#     text=str(text).lower()
    
#     text=re.sub('[^a-z]+', ' ', text)
#     tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

#     # remove stopwords
#     # stop = stopwords.words('english')
#     tokens = [token for token in tokens if token not in ext_stopwords]

#     # remove words less than three letters
#     tokens = [word for word in tokens if len(word) >= 3]

#     # lower capitalization
#     tokens = [word.lower() for word in tokens]

#     # lemmatize
# #    porter = PorterStemmer()
# #    tokens = [porter.stem(word) for word in tokens]
#     preprocessed_text= ' '.join(tokens)
#     return preprocessed_text

#%%
##########PREPROCESSING#################

# Drop unnecessary columns
columns_to_drop = ['track_id',  'track_album_id', 'playlist_name', 'playlist_id']
df = df.drop(columns=columns_to_drop)

# Convert 'track_popularity' to binary classes (0 or 1)
df['target'] = df['track_popularity'].apply(lambda x: 0 if x <= 49 else 1)

# Drop the original 'track_popularity' column
df = df.drop(columns=['track_popularity'])

# Encode categorical variables using Label Encoding
label_encoder = LabelEncoder()
df['playlist_genre'] = label_encoder.fit_transform(df['playlist_genre'])
df['playlist_subgenre'] = label_encoder.fit_transform(df['playlist_subgenre'])

# Function to detect language
def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return 'unknown'

# Apply the function to each row in the 'lyrics' column
df['detected_language'] = df['lyrics'].apply(detect_language)

# Filter out non-English rows
df = df[df['detected_language'] == 'en']

# Drop the 'detected_language' column if needed
df = df.drop(columns=['detected_language'])


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



#%%
#########SPLIT############

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




# %%
# # Convert lyrics to TF-IDF features
# tfidf_vectorizer = TfidfVectorizer(max_features=1000)
# X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['lyrics'])
# X_test_tfidf = tfidf_vectorizer.transform(X_test['lyrics'])

# # Train Logistic Regression model
# model = LogisticRegression()
# model.fit(X_train_tfidf, y_train)

# # Make predictions
# y_pred = model.predict(X_test_tfidf)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')

# # Print classification report
# print(classification_report(y_test, y_pred))


#%%

# # Example: Explore Random Forest with GridSearchCV for hyperparameter tuning
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20],
# }

# rf_model = RandomForestClassifier(random_state=42)
# grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train_tfidf, y_train)

# # Get the best model from the grid search
# best_rf_model = grid_search.best_estimator_

# # Make predictions
# y_pred_rf = best_rf_model.predict(X_test_tfidf)

# # Evaluate the model
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print(f'Random Forest Accuracy: {accuracy_rf:.2f}')
# print(classification_report(y_test, y_pred_rf))


#%%
# Selecting features and target
X = df.drop(columns=['target'])
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the lyrics using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_lyrics_tfidf = tfidf_vectorizer.fit_transform(X_train['lyrics'])
X_test_lyrics_tfidf = tfidf_vectorizer.transform(X_test['lyrics'])

# Standardize numeric features
numeric_columns = X_train.select_dtypes(include=['number']).columns
scaler = StandardScaler()
X_train_numeric_scaled = scaler.fit_transform(X_train[numeric_columns])
X_test_numeric_scaled = scaler.transform(X_test[numeric_columns])

# Concatenate TF-IDF features with numeric features
X_train_combined = pd.concat([pd.DataFrame(X_train_lyrics_tfidf.toarray()), X_train_numeric_scaled.reset_index(drop=True)], axis=1)
X_test_combined = pd.concat([pd.DataFrame(X_test_lyrics_tfidf.toarray()), X_test_numeric_scaled.reset_index(drop=True)], axis=1)

# Define models
models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression()
}

# Train and evaluate each model
results = {'Model': [], 'Accuracy': []}

for model_name, model in models.items():
    model.fit(X_train_combined, y_train)
    y_pred = model.predict(X_test_combined)
    accuracy = accuracy_score(y_test, y_pred)
    results['Model'].append(model_name)
    results['Accuracy'].append(accuracy)

# Create a DataFrame for the results
results_df = pd.DataFrame(results)

# Print the results
print(results_df)

# %%
# # Assuming 'df' is your preprocessed DataFrame
# df.to_csv('preprocessed_data.csv', index=False)
