Spotify Song Popularity Predictor
Overview
This project is focused on predicting the popularity of songs on Spotify using a rich set of features derived from both musical and lyrical attributes. The best model for this task is a sophisticated neural network, optimized through careful design and training strategies.

Prerequisites
Python 3.x
Libraries: pandas, numpy, sklearn, keras (or tensorflow), and other dependencies.
Basic understanding of machine learning concepts and Python programming.
Dataset
The dataset includes standard musical features like danceability, energy, key, loudness, and custom-processed lyrical features in a CSV file format.

Newly Created Features
Unique features from the dataset are crucial for predicting song popularity:

Syllable Counts, Variations
Novel Word Proportions
Rhyme Densities
Phonetic Similarities
Metrical Patterns Analysis
...and more.

Best Model: Neural Network with Conv1D and Dense Layers

Architecture:
Input Layer: Processes 200 features, reduced from the original dataset using PCA.
Conv1D Layer: Extracts important features from the data segments.
Flatten Layer: Converts the output of Conv1D into a one-dimensional array.
Dense Layers: Multiple layers with varying units (512, 256, 128, 100, 50) using 'relu' and 'sigmoid' activations to capture complex patterns in the data.
Output Layer: A single neuron with sigmoid activation for predicting normalized popularity scores.
Optimization and Training:
Compiled with the Adam optimizer.
Loss function: Mean Squared Error (MSE).
Includes early stopping and batch processing strategies.

Evaluation:
The model's performance is evaluated using MSE and other relevant metrics.
Usage

Data Preparation: Place your dataset in a directory and update the script with the file path.

Feature Extraction and Processing: Run the script to process and extract features.

Model Training and Evaluation: Train the model with the training set and evaluate its performance on the testing set.

Results
Outputs include the model's performance metrics and insights on its predictive accuracy.
Predictions
Apply the trained model to predict the popularity of new Spotify songs.
Notes
Ensure all necessary libraries are installed.
The model's effectiveness is subject to the quality and characteristics of the data used.
