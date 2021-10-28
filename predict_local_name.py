import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import model_from_json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd


def predict_original_name(name):
    num_words = 10000
    max_news_len = 30
    train = pd.read_csv('dataset/319/train.csv',
                        header=None,
                        names=['class', 'text'])
    # news = train['text'].astype(str)
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

    data = {'class': [1], 'text': [name]}
    test = pd.DataFrame.from_dict(data)
    test_sequences = tokenizer.texts_to_sequences(test['text'])
    x_test = pad_sequences(test_sequences, maxlen=max_news_len)

    x = np.expand_dims(x_test[0], axis=0)
    res = model_lstm.predict(x)
    argmax = np.argmax(res)

    # print("Found in %s seconds\n" % (time.time() - start_time))

    return np.max(res), argmax, classes[argmax]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\n./venv/bin/python3 predict_local_name.py \"АНАЛЬГИН ТАБ. 0.5Г N20 ОБН\"")
        exit(1)

    name = sys.argv[1]
    max_value, class_num, predicted_name = predict_original_name(name)

    print("Max Value: %s\nClass num: %s\nOrig. name: %s" % (max_value, class_num + 1, predicted_name))
