import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.layers import LSTM
import numpy as np
import sklearn as sklearn
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils
import pandas as pd
import matplotlib.pyplot as plt

num_words = 10000
max_news_len = 30
nb_classes = 100
train = pd.read_csv('dataset/100/train.csv',
                    header=None,
                    names=['class', 'title', 'text'])
news = train['text']
y_train = utils.to_categorical(train['class'] - 1, nb_classes)
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(news)
sequences = tokenizer.texts_to_sequences(news)
x_train = pad_sequences(sequences, maxlen=max_news_len)

model_lstm = Sequential()
model_lstm.add(Embedding(num_words, 32, input_length=max_news_len))
model_lstm.add(LSTM(16))
model_lstm.add(Dense(nb_classes, activation='softmax'))

model_lstm.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

model_lstm.summary()

model_lstm_save_path = 'model/100/best_model_lstm.h5'
checkpoint_callback_lstm = ModelCheckpoint(model_lstm_save_path,
                                           monitor='val_accuracy',
                                           save_best_only=True,
                                           verbose=1)

history_lstm = model_lstm.fit(x_train,
                              y_train,
                              epochs=5,
                              batch_size=128,
                              validation_split=0.1,
                              callbacks=[checkpoint_callback_lstm])

plt.plot(history_lstm.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history_lstm.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.savefig("model/100/mygraph.png")

# serialization
model_json = model_lstm.to_json()
with open("model/100/model_lstm.json", "w") as json_file:
    json_file.write(model_json)
model_lstm.save_weights("model/100/model_lstm.h5")
