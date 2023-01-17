import pickle
from flask import Flask, request, json
from flask_cors import CORS, cross_origin
from linear.linear import linear_predict
from clf.clf import clf_predict

app = Flask(__name__)

CORS(app)

def load_model():
    clf = pickle.load(open('clf/clf_model.sav', 'rb'))
    regressor = pickle.load(open('linear/linear_model.sav', 'rb'))
    return clf, regressor

classifier, regressor = load_model()

@app.route("/regress", methods=['POST'])
@cross_origin
def regress():
    data = request.get_json()

    data = [[int(ft) for ft in data.values()]]

    pred = linear_predict(data, regressor)[0] + data[0][0] * 2

    response = app.response_class(
        response=json.dumps({'grade': pred}),
        status=200,
        mimetype='application/json'
    )

    return response

@app.route("/classify", methods=['POST'])
@cross_origin
def classify():
    data = request.get_json()

    x = [int(ft) for ft in data.values()]
    pred = clf_predict(x, classifier)

    response = app.response_class(
        response=json.dumps({
            'grade': pred[0]
        }),
        status=200,
        mimetype='application/json'
    )
    return response
