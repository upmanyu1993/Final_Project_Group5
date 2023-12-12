import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def remove_outliers_with_dbscan(df, embedding_columns, eps=32, min_samples=2):
    # Step 1: Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[embedding_columns])

    # Step 2: Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_data)

    # Step 3: Identify outliers (points labeled as -1)
    outlier_indices = df.index[clusters == -1]

    # Step 4: Filter outliers
    non_outlier_df = df.drop(outlier_indices)
    return non_outlier_df
# Usage example
df = pd.read_csv('sentence_embedding.csv')  # Load your data
# df = df[[i for i in df.columns if 'track' not in i]]
embedding_columns = [i for i in df.columns if 'track' not in i]
df_no_outliers = remove_outliers_with_dbscan(df, embedding_columns)
df.to_csv('sentence_embedd.csv',index=False)


