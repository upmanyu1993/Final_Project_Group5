import pandas as pd

# Load your DataFrame
df = pd.read_csv('final_data.csv')

# List of artists
artist_names = ['Queen', 'David Guetta', 'Drake', "Guns N' Roses", 'Logic', 'The Chainsmokers', 'Martin Garrix', '2Pac', 'The Weeknd', 'Eminem']

# Specify the features
features = ['danceability', 'energy', 'key', 'mode', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

# Initialize an empty DataFrame to store the mean feature values for each artist
mean_features_df = pd.DataFrame(columns=['artist'] + features)  # Add 'artist' as the first column

# Process each artist
for artist in artist_names:
    # Filter the DataFrame for the current artist
    artist_df = df[df['track_artist'] == artist]

    # Calculate the mean of each feature for the current artist
    mean_features = artist_df[features].mean()

    # Convert Series to DataFrame with the artist's name as a column
    artist_mean_df = pd.DataFrame(mean_features).transpose()
    artist_mean_df['artist'] = artist

    # Reorder the DataFrame to have 'artist' as the first column
    artist_mean_df = artist_mean_df[['artist'] + features]

    # Concatenate the new DataFrame with the main DataFrame
    mean_features_df = pd.concat([mean_features_df, artist_mean_df], ignore_index=True)

# Save the means to a CSV file
mean_features_df.to_csv('artist_feature_means.csv', index=False)

# Print the DataFrame
print(mean_features_df)
