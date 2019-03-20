import preprocessing
import yelp_neural_networks
import svm_classification as svm

from pymongo import MongoClient
from itertools import chain

from sklearn.metrics import confusion_matrix


def error_cost(y_pred, y_val):
    cost = 0
    for i in range(0, len(y_pred)):
        if abs(y_pred[i] - y_val[i]) > 1:
            cost = cost + abs(y_pred[i] - y_val[i])
    cost = (1.0 * cost) / len(y_pred)
    return cost


def compare(a_pred, b_pred, labels):
    both_correct = 0
    both_wrong = 0
    closest_a = 0
    closest_b = 0
    same_away = 0
    a_correct = 0
    b_correct = 0
    rest = 0

    cnn_correct_indexes = []
    lstm_correct_indexes = []

    for i in range(0, len(labels)):
        if a_pred[i] == b_pred[i] and a_pred[i] == labels[i]:
            both_correct = both_correct + 1
        elif b_pred[i] != labels[i] and a_pred[i] != labels[i]:
            both_wrong = both_wrong + 1
            if abs(a_pred[i]-labels[i]) < abs(b_pred[i]-labels[i]):
                closest_a = closest_a + 1
            elif abs(a_pred[i]-labels[i]) > abs(b_pred[i]-labels[i]):
                closest_b = closest_b + 1
            else:
                same_away = same_away + 1
        elif a_pred[i] == labels[i]:
            if abs(b_pred[i] - labels[i]) > 1:
                a_correct = a_correct + 1
                cnn_correct_indexes.append(i)
        elif b_pred[i] == labels[i]:
            if abs(a_pred[i] - labels[i]) > 1:
                b_correct = b_correct + 1
                lstm_correct_indexes.append(i)
        else:
            print a_pred[i], b_pred[i]

    print 'Both: ', both_correct
    print 'None: ', both_wrong, ' but CNN get closer in ', closest_a, ' cases, while the LSTM in ', closest_b, ' cases.'
    print 'CNN: ', a_correct
    print 'LSTM: ', b_correct
    print rest

    return cnn_correct_indexes, lstm_correct_indexes



if __name__ == '__main__':

    client = MongoClient('localhost', 27017)
    db = client.Yelp

    reviews_1 = db.reviews.find({'stars': 1}, {'text': 1, 'stars': 1}).skip(100000).limit(10000)
    reviews_2 = db.reviews.find({'stars': 2}, {'text': 1, 'stars': 1}).skip(50000).limit(10000)
    reviews_3 = db.reviews.find({'stars': 3}, {'text': 1, 'stars': 1}).skip(100000).limit(10000)
    reviews_4 = db.reviews.find({'stars': 4}, {'text': 1, 'stars': 1}).skip(100000).limit(10000)
    reviews_5 = db.reviews.find({'stars': 5}, {'text': 1, 'stars': 1}).skip(100000).limit(10000)
    reviews = chain(reviews_1, reviews_2, reviews_3, reviews_4, reviews_5)

    result, stars = preprocessing.process_data(reviews, lexicon='load', using='tokenizer')
    # result_svm, stars = preprocessing.process_data(reviews, lexicon='load', using='tf-idf')


    # svm.train_model(result, stars)
    # yelp_neural_networks.train_model('lstm', result, stars)
    # yelp_neural_networks.train_model('cnn', result, stars)

    # svm.evaluate_model(result, stars)
    # yelp_neural_networks.evaluate_model('lstm', result, stars)
    # yelp_neural_networks.evaluate_model('cnn', result, stars)

    predictions_cnn = yelp_neural_networks.predict_model('cnn', result)
    predictions_lstm = yelp_neural_networks.predict_model('lstm', result)
    # predictions_svm = svm.predict_model(result_svm)

    print error_cost(predictions_cnn, stars)
    # print confusion_matrix(stars, predictions_cnn, labels=[1, 2, 3, 4, 5])

    print error_cost(predictions_lstm, stars)
    # print confusion_matrix(stars, predictions_lstm, labels=[1, 2, 3, 4, 5])

    cnn, lstm = compare(predictions_cnn, predictions_lstm, stars)

    # reviews_1 = db.reviews.find({'stars': 1}, {'text': 1, 'stars': 1}).skip(100000).limit(1000)
    # reviews_2 = db.reviews.find({'stars': 2}, {'text': 1, 'stars': 1}).skip(50000).limit(1000)
    # reviews_3 = db.reviews.find({'stars': 3}, {'text': 1, 'stars': 1}).skip(100000).limit(1000)
    # reviews_4 = db.reviews.find({'stars': 4}, {'text': 1, 'stars': 1}).skip(100000).limit(1000)
    # reviews_5 = db.reviews.find({'stars': 5}, {'text': 1, 'stars': 1}).skip(100000).limit(1000)
    # reviews = chain(reviews_1, reviews_2, reviews_3, reviews_4, reviews_5)
    #
    # samples = []
    # for review in reviews:
    #     samples.append(review['text'])
    #
    # print 'CNN'
    # for i in range(0,10):
    #     print samples[cnn[i]], predictions_cnn[cnn[i]], predictions_lstm[cnn[i]], stars[cnn[i]]
    #
    # print 'LSTM'
    # for i in range(0, 10):
    #     print samples[lstm[i]], predictions_cnn[lstm[i]], predictions_lstm[lstm[i]], stars[lstm[i]]

    # print error_cost(predictions_svm, stars)
    # print confusion_matrix(stars, predictions_svm, labels=[1, 2, 3, 4, 5])


