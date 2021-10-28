import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import model_from_json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

if __name__ == "__main__":
    num_words = 10000
    max_news_len = 30
    train = pd.read_csv('dataset/319/train.csv',
                        header=None,
                        names=['class', 'text'])
    news = train['text']

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(news)

    json_file = open('model/319/model_lstm.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_lstm = model_from_json(loaded_model_json)
    model_lstm.load_weights('model/319/best_model_lstm.h5')

    with open('dataset/319/classes.txt') as f:
        classes = f.read().splitlines()

    start_time = time.time()
    test = pd.read_csv('dataset/319/test.csv',
                       header=None,
                       names=['class', 'text'])
    test = pd.DataFrame.from_dict(test)
    test_sequences = tokenizer.texts_to_sequences(test['text'])
    x_test = pad_sequences(test_sequences, maxlen=max_news_len)

    i = 0
    fails = 0
    for x in x_test:
        x = np.expand_dims(x, axis=0)
        res = model_lstm.predict(x)
        argmax = np.argmax(res)

        if test['class'][i] != argmax + 1:
            fails += 1

            print(
                'No.%d failed: "%s"!\nExpected: "%s",\nGot: "%s",\nAccuracy: %f\n' %
                (
                    test['class'][i],
                    test['text'][i],
                    classes[i],
                    classes[np.argmax(res)],
                    np.max(res)
                )
            )
        i += 1
    print('%d fails of %s' % (fails, len(test['class'])))
