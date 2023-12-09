# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 21:18:40 2023

@author: Upmanyu
"""
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
# import plotly.express as px
from sklearn.decomposition import PCA

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Concatenate
from transformers import TFBertModel
import tensorflow as tf

def generate_model(num_numerical_features):
    # Load the pre-trained BERT model
    roberta_model = TFRobertaModel.from_pretrained('roberta-base')

    # BERT inputs
    input_ids = Input(shape=(1024,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(1024,), dtype=tf.int32, name='attention_mask')

    # BERT layer
    roberta_output = roberta_model(input_ids, attention_mask=attention_mask)[0]

    # CNN layers
    cnn_output = Conv1D(4, 6, padding='valid')(roberta_output)
    cnn_output = MaxPooling1D(1, strides=1)(cnn_output)
    cnn_output = Flatten()(cnn_output)

    # Additional input for numerical features
    numerical_input = Input(shape=(num_numerical_features,), name='numerical_data')
    numerical_dense = Dense(32, activation='relu')(numerical_input)

    # Concatenate BERT CNN output and numerical features
    combined_features = Concatenate()([cnn_output, numerical_dense])

    # Dense layers
    combined_features = Dense(128, activation='relu')(combined_features)
    # combined_features = Dropout(0.2)(combined_features)
    combined_features = Dense(8, activation='relu')(combined_features)
    # combined_features = Dropout(0.2)(combined_features)
    combined_features = Dense(4, activation='relu')(combined_features)
    # combined_features = Dropout(0.2)(combined_features)

    # Output layer
    final_output = Dense(1, activation='sigmoid')(combined_features)

    # Construct the final model
    model = Model(inputs=[input_ids, attention_mask, numerical_input], outputs=final_output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

    return model


from transformers import BertTokenizer
from transformers import TFRobertaModel, RobertaTokenizer

def preprocess_lyrics(data, tokenizer, max_length=1024):
    # Clean the lyrics
    data['lyrics_processed'] = data['lyrics.1'].str.lower()  # Convert to lowercase
    # data['lyrics_processed'] = data['lyrics_processed'].str.replace('[^a-zA-Z]', ' ', regex=True)  # Remove special characters and numbers

    # Tokenize and encode the lyrics for BERT
    tokenized_data = tokenizer(
        data['lyrics_processed'].tolist(), 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
        padding='max_length', 
        return_tensors='np'
    )

    return tokenized_data['input_ids'], tokenized_data['attention_mask']

model_name = 'roberta-base'  # Or any other variant of RoBERTa you wish to use
tokenizer = RobertaTokenizer.from_pretrained(model_name)
roberta_model = TFRobertaModel.from_pretrained(model_name)

# # Load the tokenizer
# model_name = 'bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_name)

# Load your dataset
final_data = pd.read_csv('final_data.csv')

# Further processing if needed (e.g., dropping duplicates)
final_data.drop_duplicates(['track_name', 'track_artist'], inplace=True)
final_data.drop_duplicates(['lyrics.1'], inplace=True)

# Splitting the dataset
popularity = final_data['track_popularity'].copy()
popularity = popularity/max(popularity)
x_train, x_val, y_train, y_val = train_test_split(final_data, popularity, train_size=0.8, shuffle=True)

# Prepare the text data
train_input_ids, train_attention_mask = preprocess_lyrics(x_train, tokenizer)
val_input_ids, val_attention_mask = preprocess_lyrics(x_val, tokenizer)

# Prepare the numerical data
numerical_columns = final_data.columns[12:36]  # Adjust the column indices as per your dataset
x_train_numerical = x_train[numerical_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
x_val_numerical = x_val[numerical_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

# Standardize the numerical data
scaler = StandardScaler()
x_train_numerical = scaler.fit_transform(x_train_numerical)
x_val_numerical = scaler.transform(x_val_numerical)

# Combine text data and numerical data
x_train_combined = [train_input_ids, train_attention_mask, x_train_numerical]
x_val_combined = [val_input_ids, val_attention_mask, x_val_numerical]

# Generate the model
num_numerical_features = x_train_numerical.shape[1]  # Number of numerical features
model = generate_model(num_numerical_features)
from keras.callbacks import EarlyStopping, ModelCheckpoint

model_checkpoint = ModelCheckpoint(
    'best_model.hdf5',  # Change the file path and naming as needed
    monitor='val_loss',  # or another metric that you want to monitor
    mode='min',  # for loss, 'min' mode; for accuracy, 'max' mode
    save_best_only=True,
    verbose=1
)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(x_train_combined, y_train, 
                    validation_data=(x_val_combined, y_val), 
                    epochs=100, batch_size=32, 
                    callbacks=[early_stopping, 
                               model_checkpoint])


from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
# Predicting values on the validation set
predictions = model.predict(x_val_combined)

# Calculate additional metrics
rmse = mean_squared_error(y_val, predictions, squared=False)
mape = mean_absolute_percentage_error(y_val, predictions)
r_squared = r2_score(y_val, predictions)
# Number of observations and predictors
n = len(y_val)  # Number of data points in the validation set
p = x_train_combined[0].shape[1] + x_train_combined[1].shape[1] + num_numerical_features  # Total features from BERT input, attention mask, and numerical features

# Calculate Adjusted R-Squared
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

# Print the metrics
print(f'Validation R-Squared: {r_squared}')
print(f'Validation Adjusted R-Squared: {adjusted_r_squared}')
print(f'Validation RMSE: {rmse}')
print(f'Validation MAPE: {mape}')

import matplotlib.pyplot as plt

# Predicting values on the validation set
predictions = model.predict(x_val_combined)

# Visualize the predictions against the actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_val, predictions, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')

# Save the plot to a file
plt.savefig('actual_vs_predicted.png')

# Optionally, display the plot as well
# plt.show()
