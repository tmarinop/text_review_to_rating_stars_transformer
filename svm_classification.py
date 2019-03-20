from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


def train_model(reviews, result):
    X_train, X_test, y_train, y_test = train_test_split(reviews, result, test_size=0.2, random_state=42)
    svm_classifier = OneVsRestClassifier(LinearSVC(random_state=0))
    svm_classifier.fit(X_train, y_train)
    print svm_classifier.score(X_test, y_test)
    joblib.dump(svm_classifier, './model_svm/svm_model.pkl')


def evaluate_model(reviews, result):
    X_train, X_test, y_train, y_test = train_test_split(reviews, result, test_size=0.2, random_state=42)
    svm_classifier = joblib.load('./model_svm/svm_model.pkl')
    print svm_classifier.score(X_test, y_test)


def predict_model(reviews):
    svm_classifier = joblib.load('./model_svm/svm_model.pkl')
    X_reviews = [review for review in reviews]
    predictions = svm_classifier.predict(X_reviews)
    print predictions
    return predictions

