import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

def dataset(final_data, sentence_embedding, lyrics_features):

    # Label Encoding
    label_encoder = LabelEncoder()
    final_data['track_album_release_date_encoded'] = label_encoder.fit_transform(final_data['track_album_release_date'])
    
    onehot_encoder_genre = OneHotEncoder(sparse=False)
    onehot_encoded_genre = onehot_encoder_genre.fit_transform(final_data[['playlist_genre']])
    
    onehot_encoder_subgenre = OneHotEncoder(sparse=False)
    onehot_encoded_subgenre = onehot_encoder_subgenre.fit_transform(final_data[['playlist_subgenre']])
    
    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')  # handle_unknown='ignore' to deal with unseen categories
    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    # Adding one-hot encoded data back to dataframe for 'playlist_genre'
    for i, category in enumerate(onehot_encoder_genre.categories_[0]):
        final_data[f'playlist_genre_{category}'] = onehot_encoded_genre[:, i]
    
    # Adding one-hot encoded data back to dataframe for 'playlist_subgenre'
    for i, category in enumerate(onehot_encoder_subgenre.categories_[0]):
        final_data[f'playlist_subgenre_{category}'] = onehot_encoded_subgenre[:, i]
    
    combined_cols = final_data[['track_artist']]
    onehot_encoded = onehot_encoder.fit_transform(combined_cols)
    # Creating new column names for one-hot encoded features
    new_columns = onehot_encoder.get_feature_names_out(['track_artist'])
    # Create a new DataFrame from the one-hot encoded array
    onehot_df = pd.DataFrame(onehot_encoded, columns=new_columns)
    # If final_data has an index, align the new DataFrame's index with final_data
    onehot_df.index = final_data.index
    # Concatenate the new DataFrame with the original DataFrame
    final_data = pd.concat([final_data, onehot_df], axis=1)
    
    sentence_embedding.drop('track_popularity_x',axis=1,inplace=True)
    
    sentence_embedding = sentence_embedding.rename({'track_popularity_y':'track_popularity'},axis=1)
    sentence_embedding = pd.merge(sentence_embedding, lyrics_feature, on=['track_name','track_artist','track_popularity'], how='right')
    sentence_embedding.fillna(0,inplace=True)
    final_data.drop_duplicates(['track_name','track_artist'],inplace=True)
    final_data.drop_duplicates(['lyrics.1'],inplace=True)    
    final_data.index=range(len(final_data))
    final_data  = final_data[['track_album_release_date_encoded']+(final_data.columns[806:].tolist())]
    
    sentence_embedding = pd.concat([final_data, sentence_embedding], axis=1)

    # sentence_embedding.drop('track_popularity',axis=1,inplace=True)
    
    sentence_embedding.drop(['track_artist','track_name'],axis=1,inplace=True)
    
    return sentence_embedding
        


def dataset_modelling(df, train_size, val_size):
  # Read data and drop empty target labels
      
  # Dimensionality reduction to 100 coordinates -> optimum point
    
    # print ([i for i in df.columns if 'popularity' in i])
    y = df['track_popularity'].to_numpy()
    df.drop('track_popularity', axis = 1, inplace=True)
    coors = df.dropna().values
    pca = PCA(n_components=200)
    coors = pca.fit_transform(coors)
    
    # Target definition
    # y = df['track_popularity'].to_numpy()
    # scaler = StandardScaler()
    
    print (df.shape)
    # coors = scaler.fit(coors)
    # y = scaler.transform(y.reshape(-1,1))
    y = y/np.max(y)
    # Train-Test-Val split
    x_train, x_test, y_train, y_test = train_test_split(coors, y, train_size = train_size, shuffle = True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 1 - val_size, shuffle = True)
    return x_train, y_train, x_test, y_test, x_val, y_val


"""Model design and training"""

def generate_model():
    model = Sequential()
    model.add(layers.Conv1D(6, 4, padding='valid',kernel_regularizer='l2', input_shape=(200, 1)))
    # model.add(layers.MaxPooling1D(2, strides=1))
    model.add(layers.Flatten())
     #model.add(layers.Dense(1024, activation='relu'))
       
    model.add(layers.Dense(512, activation='relu'))
     # model.add(Dropout(0.4))
    model.add(layers.Dense(256, activation='relu'))
     #model.add(Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
     #model.add(Dropout(0.3))
    model.add(layers.Dense(100, activation ='sigmoid'))
    model.add(layers.Dense(50, activation ='sigmoid'))
     #model.add(layers.Dense(25, activation = 'sigmoid'))
    model.add(layers.Dense(1, activation ='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.0001),
       loss= 'mse',
       metrics=['mae'])
    return model

def train_model(model, x_train, y_train, x_val, y_val):
    # # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=500, batch_size=64, callbacks=[early_stopping])
    
    return history


def evaluation(model, x_test, y_test):

    # Predicting values on the validation set
    predictions = model.predict(x_test)
    
    # Calculate additional metrics
    rmse = mean_squared_error(y_test, predictions, squared=False)
    mape = mean_absolute_percentage_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)
    # Number of observations and predictors
    n = len(y_test)  # Number of data points in the validation set
    p = x_train.shape[1]  # Total features from BERT input, attention mask, and numerical features
    
    # Calculate Adjusted R-Squared
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    
    # Print the metrics
    print(f'Validation R-Squared: {r_squared}')
    print(f'Validation Adjusted R-Squared: {adjusted_r_squared}')
    print(f'Validation RMSE: {rmse}')
    print(f'Validation MAPE: {mape}')
    
    return predictions


"""Save model"""
def save(model):
    model.save('lyrics_model.h5')

"""### Test and results analysis"""

def result_analysis(model, x_test, y_test):
    model.evaluate(x_test, y_test)
    
    # Predictions
    y_pred = model.predict(x_test)
    print(y_pred.shape, 'This y_pred shape')
    
    # Flatten y_pred to 1D
    y_pred_flat = y_pred.flatten()
    
    # Test dataset plot preparation
    # coors = reduce_dimensionality(x_test, 3)
    
    # Flatten y_test to 1D if it's 2D
    y_test_flat = y_test.flatten() if y_test.ndim > 1 else y_test
    
    print(y_test_flat.shape, 'This is y_test.shape')
    
    # Error difference calculation
    abs_difference = [abs(y_pred_flat[i] - y_test_flat[i]) for i in range(len(y_test_flat))]
    diff = [y_pred_flat[i] - y_test_flat[i] for i in range(len(y_test_flat))]
    
    # Create DataFrame
    aux = pd.DataFrame({
        'y_pred': y_pred_flat,
        'y_test': y_test_flat,
        'diff': diff,
        'abs_diff': abs_difference
    })
    print (aux.loc[aux['y_test']==0,'y_pred'].describe())
    return aux


def plot_residuals(model, x_test, y_test):

    y_pred = model.predict(x_test)
    # Calculate residuals
    residuals = y_test.flatten() - y_pred.flatten()
    
    # Residual vs. Fitted Plot
    plt.scatter(y_pred, residuals)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residual vs. Fitted')
    plt.axhline(y=0, color='grey', linestyle='dashed')
    plt.savefig('residual_fitted_values')
    # plt.show()
    
    # Histogram of Residuals
    plt.hist(residuals, bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.savefig('Histogram of Residuals')
    # plt.show()
    
    # Q-Q plot
    sm.qqplot(residuals, line ='45')
    plt.title('Normal Q-Q Plot')
    plt.savefig('Normal Q-Q Plot')
    # Durbin-Watson Test
    dw_statistic = durbin_watson(residuals)
    print(f'Durbin-Watson statistic: {dw_statistic}')

final_data = pd.read_csv('final_data.csv')
sentence_embedding = pd.read_csv('sentence_embedding.csv')
lyrics_feature = pd.read_csv('lyrics_feature.csv')

df = dataset(final_data, sentence_embedding, lyrics_feature)
x_train, y_train, x_test, y_test, x_val, y_val = dataset_modelling(df, 0.8, 0.1)
model = generate_model()
history = train_model(model, x_train, y_train, x_val, y_val)
pred = evaluation(model, x_test, y_test)
aux = result_analysis(model, x_test, y_test)
plot_residuals(model, x_test, y_test)

