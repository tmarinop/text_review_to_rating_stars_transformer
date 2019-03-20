from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.text import Tokenizer as Keras_Tokenizer
from nltk.corpus import stopwords
from sklearn.externals import joblib
import re
import json
import time

keras_tokenizer = Keras_Tokenizer(filters='')

# Stemming didn't help on NN implementations, so Porter Stemmer has been removed from latest release.


def remove_stop_words(text):
    """
    Takes a string and removes stopwords.
    Stopwords are retrieved from nltk. Then some words that are necessary in sentiment analysis are removed from the
    stopwords.
    :param text: A string.
    :return filtered_tokens: A list of words.
    """
    useful_words = {"shouldn't", "wasn", "needn't", "hasn't", "isn't", "just", "aren", "aren't", "couldn't", "wouldn't",
                    "haven't", "doesn", "weren", "don", "mustn't", "more", "not", "didn't", "hadn", "weren't", "don't",
                    "doesn't", "hasn", "very", "won", "wouldn", "most", "haven", "didn", "against", "won't", "needn",
                    "wasn't"}
    stop_words = set(stopwords.words('english')) - useful_words


    kept_words = [i.encode("ascii") for i in text.lower().split() if i not in stop_words]
    return ' '.join(kept_words)


def replace_digits(text):
    """
    Takes a string and replaces integers with '#'.
    :param text: A string.
    :return A list of the tokens.
    """
    return re.sub('\d+', '', text)


def remove_punctuation(text):
    """
    Removes punctuation from string
    :param text: A string
    :return: A string stripped by punctuation
    """
    return re.sub('[^a-zA-Z# ]', '', text)


def convert_to_int(wrd2id, text):
    """
    Converts word tokens from string to int using the word2id dictionary.
    :param text: text as a string to be translated
    :param wrd2id: vocabulary that translates word to unique ids
    :return: list of int representing the initial text
    """
    keras_tokenizer.word_index = wrd2id
    token_seq_int = keras_tokenizer.texts_to_sequences(text.split())
    return [token for sublist in token_seq_int for token in sublist]


def process_data(mongoCursor, lexicon='load', using='tf-idf'):
    """
    Removes stopwords, punctuation and digits and then maps each word to a unique integer value.
    :param mongoCursor: input data inserted from mongo
    :param lexicon: whether to load or to create word-to-int lexicon from data
    :return: Processed input
    """

    min_word_df = 2
    vocabulary_size = 80000

    start_time = time.time()

    text_data = []
    stars_data = []
    for review in mongoCursor:
        text = review['text']
        text_no_digits = replace_digits(text)
        text_no_punctuation = remove_punctuation(text_no_digits)
        clean_text = remove_stop_words(text_no_punctuation)
        text_data.append(clean_text)
        stars_data.append(review['stars'])

    if using == 'tokenizer':
        if lexicon == 'load':
            print 'loading word index...'
            with open('data/word_index.json', 'r') as fp:
                wrd2id = json.load(fp)
                print('Found %s unique tokens' % len(wrd2id))
        else:
            print 'creating word index...'
            count_vectorizer = CountVectorizer(input='content', analyzer='word', max_df=1.0, min_df=min_word_df,
                                               max_features=vocabulary_size)
            count_vectorizer.fit_transform(text_data)
            vocabulary = count_vectorizer.vocabulary_
            wrd2id = dict((w, i + 1) for i, w in enumerate(vocabulary))
            print('Found %s unique tokens' % len(wrd2id))
            print 'writting indexes...'
            with open('data/word_index.json', 'w') as fp:
                json.dump(wrd2id, fp)

        processed_data = []
        for text in text_data:
            processed_data.append(convert_to_int(wrd2id, text))

        end_time = time.time()
        print "--- Process time: %s seconds ---" % (end_time - start_time)
        return processed_data, stars_data

    elif using == 'tf-idf':
        if lexicon == 'load':
            vectorizer = joblib.load('./data/tf-idf.pkl')
            X = vectorizer.transform(text_data)
        else:
            vectorizer = TfidfVectorizer(min_df=min_word_df, ngram_range=(1,2))
            X = vectorizer.fit_transform(text_data)
            joblib.dump(vectorizer, './data/tf-idf.pkl')
        end_time = time.time()
        print "--- Process time: %s seconds ---" % (end_time - start_time)
        return X, stars_data


def process_one(text):
    """
    Removes stopwords, punctuation and digits and then maps each word to a unique integer value.
    :param text: one input string
    :return: Processed input
    """
    start_time = time.time()

    text_data = []
    text_no_digits = replace_digits(text)
    text_no_punctuation = remove_punctuation(text_no_digits)
    clean_text = remove_stop_words(text_no_punctuation)
    text_data.append(clean_text)

    print 'loading word index...'
    with open('data/word_index.json', 'r') as fp:
        wrd2id = json.load(fp)
        print('Found %s unique tokens' % len(wrd2id))

    processed_data = []
    for text in text_data:
        processed_data.append(convert_to_int(wrd2id, text))

    end_time = time.time()

    print "--- Process time: %s seconds ---" % (end_time - start_time)

    return processed_data
