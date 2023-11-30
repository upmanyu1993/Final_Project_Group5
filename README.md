# Final_Project_Group5

Spotify Song Popularity Predictor

Overview
This project aims to predict the popularity of songs on Spotify using a unique set of features derived from both musical and lyrical attributes. The prediction model is built using an XGBoost regression framework and optimized through grid search for hyperparameter tuning.

Prerequisites
Python 3.x
Libraries: pandas, numpy, sklearn, xgboost, nltk, tqdm (for progress bars), and any other dependencies required for these libraries.
Basic understanding of machine learning concepts and Python programming.
Dataset
The dataset is a CSV file containing features of Spotify songs. It includes standard musical features like danceability, energy, key, and loudness, as well as custom-processed lyrical features.

Newly Created Features
The script processes and creates several unique features from the dataset that are pivotal in predicting song popularity. These features include:

Syllable Count per Line: The number of syllables in each line of a song's lyrics.
Syllable Count per Word: Average syllable count in the words of the song.
Syllable Variation: The variability in the number of syllables per line across the song.
Novel Word Proportion: The proportion of new words in each line compared to the previous line.
Rhymes per Line: Number of rhyming words in each line.
Rhymes per Syllable: Average number of rhymes per syllable in the song.
Rhyme Density: Proportion of syllables in the song that are part of a rhyme.
End Pairs per Line: Frequency of rhymes at the end of lines.
End Pairs Variation: Analysis of the change in syllable counts between rhyming lines.
Average End Score: A metric representing the phonetic similarity of end rhymes.
Meter Analysis: Classification of lines into metrical patterns such as iambic, trochaic, etc.
These features were developed to capture the nuanced aspects of music and lyrics that influence a song's popularity.

Model
An XGBoost regression model predicts song popularity based on these features. The model uses grid search to optimize its hyperparameters, aiming to achieve the best performance.

Usage
Data Preparation:

Ensure your dataset CSV file is placed in an accessible directory.
Update the file path in the script to point to your dataset.
Feature Extraction and Processing:

The script extracts and processes both musical and newly created lyrical features from the dataset.
Model Training and Evaluation:

The dataset is split into training and testing sets.
The XGBoost model is trained on the training set and evaluated on the testing set using mean squared error (MSE).
Results:

The script provides the best hyperparameters, the best grid search score, and the MSE on the test set.
These outputs help evaluate the model's predictive performance.
Predictions:

The trained model can be applied to predict the popularity of new songs.
Notes
Ensure all required libraries are installed before running the script.
Hyperparameter grids can be adjusted as needed for different model explorations.
The effectiveness of the model may vary based on the dataset's quality and characteristics.

