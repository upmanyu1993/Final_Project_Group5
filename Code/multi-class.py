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
















# Now, X_train, X_val, X_test are your train_features, val_features, and test_features respectively
# train_features, val_features, test_features = X_train, X_val, X_test

# batch_size = 64
# train_loader = prepare_reduced_data_loader(X_train, y_train, batch_size)
# val_loader = prepare_reduced_data_loader(X_val, y_val, batch_size)
# test_loader = prepare_reduced_data_loader(X_test, y_test, batch_size)

# # Determine the size of the PCA-reduced feature set
# input_size = reduced_features.shape[1]

# # Create an instance of the CustomClassifier
# model = CustomClassifier(input_size, num_classes)


# # Train and evaluate the model as before
# # Use train_model and evaluate_model functions with the modifications previously discussed
# # Assuming model is already defined
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# # Define the loss function and optimizer
# loss_function = nn.CrossEntropyLoss()
# optimizer = Adam(model.parameters(), lr=1e-5)  # Adjust learning rate as needed


# def validate_model(model, val_loader, device):
#     model.eval()  # Set the model to evaluation mode
#     val_loss = 0.0
#     all_predictions = []
#     all_labels = []

#     with torch.no_grad():  # No need to track gradients during validation
#         for batch in val_loader:
#             features, labels = batch
#             features, labels = features.to(device), labels.to(device)

#             outputs = model(features)
#             loss = loss_function(outputs, labels)
#             val_loss += loss.item()

#             preds = torch.argmax(outputs, axis=1)
#             all_predictions.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     accuracy = accuracy_score(all_labels, all_predictions)
#     f1 = f1_score(all_labels, all_predictions, average='weighted')

#     print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {accuracy}, F1 Score: {f1}')
#     return accuracy, f1

# from tqdm import tqdm
# def train_model(model, train_loader, val_loader, device, epochs):
#     best_f1_score = 0.0

#     for epoch in tqdm(range(epochs)):
#         model.train()  # Set the model to training mode
#         running_loss = 0.0

#         for batch in train_loader:
#             features, labels = batch
#             features, labels = features.to(device), labels.to(device)

#             optimizer.zero_grad()  # Clear gradients
#             outputs = model(features)
#             loss = loss_function(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss/len(train_loader)}')

#         # Validation step
#         val_accuracy, val_f1 = validate_model(model, val_loader, device)

#         # Update the best F1 score
#         if val_f1 > best_f1_score:
#             best_f1_score = val_f1
#             # Save the model
#             torch.save(model.state_dict(), 'best_model.pth')

#     print(f'Best F1 Score: {best_f1_score}')

# from sklearn.metrics import classification_report

# def evaluate_model(model, data_loader, device):
#     model.eval()  # Set the model to evaluation mode

#     all_predictions = []
#     all_labels = []
    
#     with torch.no_grad():  # No gradient needed
#         for batch in data_loader:
#             inputs = batch[0].to(device)
#             labels = batch[1].to(device)

#             outputs = model(inputs)
#             _, predictions = torch.max(outputs, 1)

#             all_predictions.extend(predictions.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     return all_labels, all_predictions


# epochs = 100  # Define the number of epochs
# train_model(model, train_loader, val_loader, device, epochs)
# # Call the evaluate function
# true_labels, predictions = evaluate_model(model, test_loader, device)

# # Generate the classification report
# report = classification_report(true_labels, predictions)
# print("Classification Report:\n")
# print(report)
