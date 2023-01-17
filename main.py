from flask import Flask, request, json
from flask_cors import CORS
from linear.linear import linear_predict
from clf.clf import clf_predict

app = Flask(__name__)
CORS(app)

@app.route("/regress", methods=['POST'])
def regress():
    data = request.get_json()


    data = [[int(ft) for ft in data.values()]]

    pred = linear_predict(data)[0] + data[0][0] * 2

    response = app.response_class(
        response=json.dumps({'grade': pred}),
        status=200,
        mimetype='application/json'
    )

    return response

@app.route("/classify", methods=['POST'])
def classify():
    data = request.get_json()

    x = [int(ft) for ft in data.values()]
    pred = clf_predict(x)
    print(pred)

    response = app.response_class(
        response=json.dumps({'grade':pred[0]}),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == "__main__":
    app.run(port=8080)