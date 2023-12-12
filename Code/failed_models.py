from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN, Input, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

def generate_model():
    model = Sequential()
    model.add(Input(shape=(819,)))  # Input layer
    model.add(Dense(512, activation='relu',kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))  # Added L2 regularization
    model.add(Reshape((512, 1)))  # Reshaping for LSTM
    model.add(LSTM(128, return_sequences=True))  # Reduced units and added regularization
    model.add(Dropout(0.2))
    model.add(LSTM(64))  # Reduced units and added regularization
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for regression

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model

def generate_model():
    model = Sequential()
    model.add(Input(shape=(819,)))  # Input layer
    model.add(Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))  # Dense layer with regularization
    model.add(Reshape((512, 1)))  # Reshaping for RNN/LSTM

    # RNN layer
    model.add(SimpleRNN(128, return_sequences=True))  # RNN layer

    # LSTM layer
    model.add(LSTM(64, return_sequences=True))  # LSTM layer
    model.add(Dropout(0.2))

    # Another LSTM layer
    model.add(LSTM(32))  # LSTM layer with fewer units
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))  # Output layer for regression

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model

from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.utils import plot_model
import tensorflow as tf
def add_cnn_layer(bert_output):
    # Reshape for CNN
    reshaped_output = tf.keras.layers.Reshape((512, 819, 1))(bert_output)
    cnn_output = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(reshaped_output)
    cnn_output = tf.keras.layers.Flatten()(cnn_output)
    return cnn_output

def generate_model():
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    bert_model = TFBertModel.from_pretrained(model_name)

    # Define the model architecture
    input_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(512,), dtype=tf.int32, name='attention_mask')

    # Get the embeddings from BERT
    bert_output = bert_model(input_ids, attention_mask=attention_mask)
    cnn_output = add_cnn_layer(bert_output[0])  # Using the last hidden state for CNN

    # Add a dense layer as the output layer
    output = tf.keras.layers.Dense(1, activation='sigmoid')(cnn_output)

    # Construct the final model
    model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def generate_model(input_shape):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=input_shape))
    # model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # or another activation, depending on your task

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and compile the model
model = generate_model((x_train.shape[1],))  # Adjust the input shape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, SimpleRNN, LSTM, Dropout, Activation, Permute, Multiply, Lambda, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import tensorflow as tf

def attention_mechanism(inputs):
    # inputs.shape = (batch_size, time_steps, lstm_units)
    time_steps = inputs.shape[1]
    lstm_units = inputs.shape[2]

    # Attention mechanism
    a = Dense(1, activation='tanh')(inputs)
    a = Flatten()(a)
    a = Activation('softmax')(a)
    a = Reshape((time_steps, 1))(a)
    output_attention = Multiply()([inputs, a])
    return output_attention

def generate_model():
    input_layer = Input(shape=(817,))
    x = Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(input_layer)
    x = Reshape((512, 1))(x)

    # RNN layer
    x = SimpleRNN(512, return_sequences=True)(x)

    # LSTM layer
    lstm_out = LSTM(32, return_sequences=True)(x)
    x = Dropout(0.2)(lstm_out)

    # Attention layer
    attention_out = attention_mechanism(lstm_out)

    # Another LSTM layer
    lstm_out2 = LSTM(16, return_sequences=False)(attention_out)
    x = Dropout(0.2)(lstm_out2)

    # Output layer
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model
