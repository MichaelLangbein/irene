from flask import Flask, jsonify, request
from modelPrediction import predictSingleBatch
from radarData import getSingleBatch


app = Flask(__name__)


@app.route("/getRadarVideo", methods=["GET"])
def getRadarVideo():
    fromTime = request.args.get("fromTime")
    toTime = request.args.get("toTime")
    data = getSingleBatch(fromTime, toTime)
    return jsonify(data)


@app.route("/predict", methods=["GET"])
def predict():
    result = predictSingleBatch()
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
