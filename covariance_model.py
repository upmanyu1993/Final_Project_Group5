import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import re
import nltk
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D,MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import metrics
from keras import backend as K
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import load_model
import tensorflow as tf
import requests
from sklearn.preprocessing import StandardScaler
stop_words = stopwords.words('english')
stopwords_list = requests.get("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt").content
ext_stopwords = list(set(stopwords_list.decode().splitlines()))

stop_words.extend(ext_stopwords)

es = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=1, mode='auto')
mc = ModelCheckpoint('covariance_final_project.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

GLOVE_6B_100D_PATH = "glove.6B.100d.txt"
encoding='utf-8'

#reading traing and test data

df = pd.read_csv('data_model.csv')
df = df[df['language']=='en']

df.loc[df['track_popularity']>=50,'popularity'] = 1
df.loc[df['track_popularity']<50,'popularity'] = 0

def glove(total):
    """
    Glove word vector is read and vectors is given to each vector and dictionary
    is returned where all the words in corpus as key with their vector as value.
    Parameter:
        total(dict): dictionary containing all the words
    Return
        dictionary of words with their wordvectors
    """
    glove_small = {}
    all_words = set(w for words in total for w in words)
    with open(GLOVE_6B_100D_PATH, "rb") as infile:
        for line in infile:
            parts = line.split()
            word = parts[0].decode(encoding)
            if (word in all_words):
                nums=np.array(parts[1:], dtype=np.float32)
                glove_small[word] = nums
        infile.close()
    return glove_small

# from ftfy import fix_text
# def preprocessing(string, n=3):
#     string = fix_text(string) # fix text encoding issues
#     string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
#     string = string.lower() #make lower case
#     chars_to_remove = [")","(",".","|","[","]","{","}","'"]
#     rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
#     string = re.sub(rx, '', string) #remove the list of chars defined above
#     string = string.replace('&', 'and')
#     string = string.replace(',', ' ')
#     string = string.replace('-', ' ')
#     string = string.title() # normalise case - capital at start of each word
#     string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single space
#     string = ' '+ string +' ' # pad names for ngrams...
#     string = re.sub(r'[,-./]|\sBD',r'', string)
#     ngrams = zip(*[string[i:] for i in range(n)])
#     return [''.join(ngram) for ngram in ngrams]

def preprocessing(text):
    """
    It takes the string and preprocess the string with the help pf nltk library
    Parameters:
        text(str): string which needs to be prerprocessed
    Return
        preprocessed string with no stopwords
    """
    # from nltk.corpus import stopwords
    text=str(text).lower()
    
    text=re.sub('[^a-z]+', ' ', text)
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    # remove stopwords
    # stop = stopwords.words('english')
    tokens = [token for token in tokens if token not in ext_stopwords]

    # remove words less than three letters
    tokens = [word for word in tokens if len(word) >= 3]

    # lower capitalization
    tokens = [word.lower() for word in tokens]

    # lemmatize
#    porter = PorterStemmer()
#    tokens = [porter.stem(word) for word in tokens]
    preprocessed_text= ' '.join(tokens)
    return preprocessed_text

class MeanEmbeddingVectorizer(object):

    """
    Class simply returns the vector of word

    """
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = 100

    def fit(self, X):
        return self

    def transform(self, X):
        return np.array([([self.word2vec[w] for w in words if w in self.word2vec])
            for words in X
        ])

def list_lists(filing):
    """
    Spliting the article in words
    Parameter:
        filing(series/list): contains all the filings having 10Q/K filings with mda extracted
    Return :
        List of lists of words
    """
    test=[]
    for i in filing:
        i=preprocessing(i)
        i=i.split()
        test.append(i)
    return test

def get_covariance_vector(article,w2v,n):
    """
    Get Covariance vectors which is calculated by transposing word vectors and calculating
    By numpy and taking upper triangular matrix as vector
    Parameter:
        article(list): list of lists of words
        w2v: object from Meanembeddingvectoriser
        n(int): number of articles
    Return
        numpy vector
    """
    covariance_vector=[]
    for i in article:
    #    i = train_article[4]
        word_vector = w2v.transform([i])
        word_vector = word_vector.reshape(word_vector.shape[1],100)
        word_vector_transpose = word_vector.transpose()
        covariance_matrix = np.cov(word_vector_transpose)
        covariance_matrix = covariance_matrix.astype(np.float16, copy=False)
        upper_tri_covar_matrix = covariance_matrix[np.triu_indices(100)]
        covariance_vector.append(upper_tri_covar_matrix)
    covariance_vector=np.array(covariance_vector,dtype='float16')
    covariance_vector=covariance_vector.reshape(n,5050,1)

    return covariance_vector

def check_model(model,x,y,x_test,y_test,es,mc):
    """
    Takes model and training set and test set and check the model by taking look on val_loss
    if it is decreasing or not
    """
    model.fit(x,y,batch_size=8,epochs=100,verbose=1,validation_data=(x_test, y_test),callbacks=[es,mc])

def build():
    """
    Building model using deep nuetral network using keras
    """
    model = Sequential()
    # model.add(Dense(4096,
    #                  activation='relu',
    #                  input_shape=(5050,1)))
    # model.add(Dense(2048,
    #                  activation='relu'))
    model.add(Dense(1024,
                     activation='relu',input_shape=(5050,1)))
#    model.add(Dropout(0.7))
    model.add(Dense(512,
                     activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(256,
                     activation='relu'
                     ))
    model.add(Dense(128,
                     activation='relu'
                     ))
    model.add(Dense(64,
                     activation='relu'
                     ))
    model.add(Dense(32,
                     activation='relu'
                     ))
    model.add(Dense(8,
                     activation='relu'
                     ))
#    model.add(Dropout(0.4))
#    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001,decay_steps = 100, decay_rate=0.0000001)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    # sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc',metrics.binary_accuracy])
    return model

def main():
    """
    Function where all the small-functions is used
    """
    
    article = list_lists(df['lyrics'])
    
    # article = list_lists(X_train)
    w2v_train = glove(article)
    w2v=MeanEmbeddingVectorizer(w2v_train).fit(article)
    X = get_covariance_vector(article,w2v,len(article))
    
    categorical_cols = ['playlist_genre', 'playlist_subgenre']
    temp_df = pd.get_dummies(df, columns=categorical_cols)
    
    arr2_reshaped = np.reshape(temp_df.iloc[:,24:].values, (15405, 30, 1))
    X = np.concatenate((X, arr2_reshaped), axis=1)
    
    reshaped = np.reshape(df.iloc[:,12:24].values, (15405, 12, 1))
    X = np.concatenate((X, reshaped), axis=1)
    
    scaler = StandardScaler()

    X_2d = X.reshape(-1, X.shape[1])

    scaler = StandardScaler()
    scaled_data_2d = scaler.fit_transform(X_2d)
    X = scaled_data_2d.reshape(len(article), X.shape[1], 1)
    # scaler.fit(X_train)
    X_train, X_test, y_train, y_test = train_test_split(X, df['popularity'],test_size=0.2,train_size=0.8)
    # X_train_scaled = scaler.transform(X_train)
    # X_test_scaled = scaler.transfrom(X_test)
    
    m = build()
    check_model(m,X_train,y_train,X_test,y_test,es,mc)
    m = load_model('/content/sample_data/covariance_final_project.h5')
    # df_val=pd.read_csv('validation_data.csv')
    # val_article = list_lists(df_val['filing'])
    # w2v_train = glove(val_article)
    # w2v=MeanEmbeddingVectorizer(w2v_train).fit(val_article)
    # X_val = get_covariance_vector(val_article,w2v,len(val_article))
    # y_score = m.predict(X_val)
    # y_score = (y_score > 0.60)

    # cm = confusion_matrix(df_val['label'], y_score)
    y_score_test = m.predict(X_test)
    y_score_test = (y_score_test>0.50)
    # cm1 = confusion_matrix(y_test, y_score_test)
    report_test=classification_report(y_test,y_score_test)
    print (report_test)
    # report_validation = classification_report(df_val['label'],y_score)

main()
