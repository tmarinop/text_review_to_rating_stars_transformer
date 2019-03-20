import numpy as np
import json
import time
import attention
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Convolution1D, Activation, SpatialDropout1D, LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import GlobalMaxPooling1D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
from keras import regularizers
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

embedding_dim = 200

glove_path = '/Users/Giannis/Downloads/glove.6B/glove.6B.200d.txt'
STAMP = 'yelp_reviews'


def load_wrd_embeddings():
    json_file = open('data/word_index.json')
    json_string = json_file.read()
    wrd2id = json.loads(json_string)
    vocab_size = len(wrd2id)
    print "Found %s words in the vocabulary." % vocab_size

    embedding_idx = {}
    glove_f = open(glove_path)
    for line in glove_f:
        values = line.split()
        wrd = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_idx[wrd] = coefs
    glove_f.close()
    print "Found %s word vectors." % len(embedding_idx)

    embedding_mat = np.zeros((vocab_size + 1, embedding_dim))
    for wrd, i in wrd2id.items():
        embedding_vec = embedding_idx.get(wrd)
        # words without embeddings will be left with zeros.
        if embedding_vec is not None:
            embedding_mat[i] = embedding_vec

    print embedding_mat.shape
    return embedding_mat, vocab_size


def build_net_lstm(review_len, embedding_dims):
    embedding_mat, vocab_size = load_wrd_embeddings()

    print('Build model_lstm...')
    model = Sequential()

    model.add(Embedding(input_dim=vocab_size + 1,
                        output_dim=embedding_dims,
                        input_length=review_len,
                        weights=[embedding_mat],
                        trainable=False))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(128, name='blstm_1',
                            activation='tanh',
                            recurrent_activation='hard_sigmoid',
                            recurrent_dropout=0.0,
                            dropout=0.3,
                            kernel_initializer='glorot_uniform',
                            return_sequences=True),
                            merge_mode='concat'))
    model.add(BatchNormalization())
    # model.add(GlobalMaxPooling1D())
    model.add(attention.AttentionWithContext())
    model.add(Dropout(0.4))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    return model


def build_net_cnn(review_len, embedding_dims):
    embedding_mat, vocab_size = load_wrd_embeddings()

    print('Build model_cnn...')
    model = Sequential()

    model.add(Embedding(input_dim=vocab_size + 1,
                        output_dim=embedding_dims,
                        input_length=review_len,
                        weights=[embedding_mat],
                        trainable=False))
    model.add(SpatialDropout1D(0.2))
    model.add(Convolution1D(128, 2,
              activation='relu',
              padding='valid',
              kernel_initializer='lecun_uniform',
              kernel_regularizer=regularizers.l2(0.0)))
    model.add(BatchNormalization())
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.4))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    return model


def train_model(nn, reviews, result):
    max_review_len = 100

    X_reviews = [review for review in reviews]
    y_data = [int(star) for star in result]

    y_data_new = []

    starmap = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    for y in y_data:
        y_data_new.append(starmap[y])

    X_reviews_train, X_reviews_val, y_train, y_val = train_test_split(X_reviews, y_data_new,
                                                                      test_size=0.2,
                                                                      random_state=10)
    y_train = to_categorical(y_train, 5)
    y_val = to_categorical(y_val, 5)

    print('Average train review sequence length: {}'.format(np.mean(list(map(len, X_reviews_train)), dtype=int)))
    X_reviews_train = pad_sequences(X_reviews_train,
                                    maxlen=max_review_len,
                                    padding='post',
                                    truncating='post',
                                    dtype='float32')

    X_reviews_val = pad_sequences(X_reviews_val,
                                  maxlen=max_review_len,
                                  padding='post',
                                  truncating='post',
                                  dtype='float32')

    print X_reviews_train.shape

    if nn == 'cnn':
        model = build_net_cnn(max_review_len, embedding_dim)

        model_json = model.to_json()
        with open("model_cnn/" + STAMP + ".json", "w") as json_file:
            json_file.write(model_json)

        early_stopping = EarlyStopping(monitor='val_acc', patience=5)
        bst_model_path = "model_cnn/" + STAMP + '.h5'

    elif nn == 'lstm':
        model = build_net_lstm(max_review_len, embedding_dim)

        model_json = model.to_json()
        with open("model_lstm_with_attention/" + STAMP + ".json", "w") as json_file:
            json_file.write(model_json)

        early_stopping = EarlyStopping(monitor='val_acc', patience=5)
        bst_model_path = "model_lstm_with_attention/" + STAMP + '.h5'

    model_checkpoint = ModelCheckpoint(bst_model_path,
                                       monitor='val_acc',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    if nn == 'cnn':
        epochs = 20
    elif nn == 'lstm':
        epochs = 10

    s_time = time.time()
    model.fit(X_reviews_train, y_train,
              validation_data=(X_reviews_val, y_val),
              epochs=epochs,
              batch_size=128,
              shuffle=True,
              callbacks=[early_stopping, model_checkpoint],
              verbose=1)
    e_time = time.time()

    print 'training time:', (e_time - s_time)

    if nn == 'cnn':
        model.load_weights('model_cnn/' + STAMP + '.h5')
    elif nn == 'lstm':
        model.load_weights('model_lstm/' + STAMP + '.h5')

    predictions = model.predict(X_reviews_val)

    predictions = [(np.argmax(pr)+1) for pr in predictions]
    y_val = [(np.argmax(y)+1) for y in y_val]

    print predictions[:20]
    print y_val[:20]


def load_model(nn):
    """
    Loads the trained model_cnn and weights.
    :param nn: whether the model to be loaded is 'lstm' or 'cnn'
    :return the loaded model
    """
    # Load the model_cnn architecture.
    json_file = open('model_' + nn + '/' + STAMP + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model_cnn
    loaded_model.load_weights('model_' + nn + '/' + STAMP + '.h5')
    print("Loaded model_cnn from disk")
    return loaded_model


def evaluate_model(nn, reviews, result):
    max_review_len = 100

    X_reviews = [review for review in reviews]

    y_data = [int(star) for star in result]
    y_data_new = []
    starmap = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    for y in y_data:
        y_data_new.append(starmap[y])

    X_reviews_train, X_reviews_val, y_train, y_val = train_test_split(X_reviews, y_data_new,
                                                                    test_size=0.2,
                                                                    random_state=10)

    y_val = to_categorical(y_val, 5)

    print('Average train review sequence length: {}'.format(np.mean(list(map(len, X_reviews_train)), dtype=int)))

    X_reviews_train = pad_sequences(X_reviews_train,
                                    maxlen=max_review_len,
                                    padding='post',
                                    truncating='post',
                                    dtype='float32')

    X_reviews_val = pad_sequences(X_reviews_val,
                                  maxlen=max_review_len,
                                  padding='post',
                                  truncating='post',
                                  dtype='float32')

    print X_reviews_train.shape

    model = load_model(nn)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    predictions = model.evaluate(X_reviews_val, y_val)
    print predictions


def predict_model(nn, reviews):
    max_review_len = 100

    X_reviews = [review for review in reviews]

    print('Average train review sequence length: {}'.format(np.mean(list(map(len, X_reviews)), dtype=int)))

    X_reviews = pad_sequences(X_reviews,
                              maxlen=max_review_len,
                              padding='post',
                              truncating='post',
                              dtype='float32')

    print X_reviews.shape

    model = load_model(nn)

    predictions = model.predict(X_reviews)
    predictions = [(np.argmax(pr) + 1) for pr in predictions]

    return predictions
