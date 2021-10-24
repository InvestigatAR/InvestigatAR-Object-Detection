from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
from yolo import process
from datetime import datetime
from random import randint
import json


app = Flask(__name__)
CORS(app)
# uploads_dir = os.path.join(app.instance_path, 'uploads')
# output_dir = os.path.join(app.instance_path, 'output')


@app.route('/predict/', methods=['POST',])
def predict_b64():
    json_data = request.get_json() #Get the POSTed json
    dict_data = json_data #Convert json to dictionary

    img = dict_data["b64"] #Take out base64# stre)
    # print(x)
    # b64 = x['b64']

    
    objects_count, objects_confidence = process(img)

    response = {
        'objects_count': objects_count, 
        'objects_confidence': objects_confidence
    }

    return jsonify({"data": response}), 200


if __name__ == '__main__':
    app.run(host="localhost", port=8080)
