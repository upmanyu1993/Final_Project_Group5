import pandas as pd

# Load dataset
df = pd.read_csv('final_data.csv')

# Check if DataFrame is loaded properly
print("DataFrame Loaded. Number of Rows:", df.shape[0], "Number of Columns:", df.shape[1])

# Print first few rows to check data
print("First Few Rows:")
print(df.head())

# Check for Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Splitting into 3 Classes based on 'track_popularity'
df['target'] = df['track_popularity'].apply(lambda x: 0 if x <= 33 else (1 if x <= 66 else 2))
print("\nDescriptive Statistics for 'target':")
print(df['track_popularity'].describe())

# Count of each class
class_counts = df['target'].value_counts()
print("\nClass Counts:")
print(class_counts)
