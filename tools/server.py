from flask import Flask, make_response, jsonify, request
import yelp_neural_networks
import preprocessing
import numpy as np

app = Flask(__name__)


def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]


@app.route('/')
def index():
    return "Welcome!"


@app.route('/reviews/api/v1.0/predict/', methods=['GET'])
def get_prediction():
    text = request.args.get('text')
    return make_response(jsonify({'text': text,
                                  'rating': yelp_neural_networks.predict_model('lstm',
                                                                               preprocessing.process_one(text))[0]}))


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    app.run(debug=False)
