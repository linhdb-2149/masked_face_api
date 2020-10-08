from flask import Flask, request, jsonify
from io import BytesIO
import numpy as np
import requests
import base64
from helpers import utils
from helpers.grpc_model import MaskClassifier, Detector
from helpers import configs
import json
import time

app = Flask(__name__)

classifier = MaskClassifier()
face_dtt = Detector()

@app.route('/')
@app.route('/home/', methods=['GET', 'POST'])
def home():
    return "Home"

@app.route('/mask_detect', methods=['POST'])
def mask_detect():
    try:
        encoded_image = request.form['image']
        if encoded_image.startswith("data:image"):
            encoded_image = encoded_image.split(",")[1]
        img_arr = utils.decode(encoded_image)
    except Exception:
        return utils.json_format(
                code=500, 
                data={},
                message='No image',
                errors={
                }
            )
    
    (faces, bboxes) = face_dtt.predict(img_arr)

    data_return = list()
    for i in range(len(faces)):
        score = classifier.predict(faces[i])
        if score > configs.CONFIDENCE_THRESHOLD:
            label = 'Mask'
        else:
            label = 'No mask'

        data_return.append(
            {
                'label': str(label),
                'score': float(score), 
                'bbox':
                {
                    'xmin': bboxes[i][0], 
                    'xmax': bboxes[i][1], 
                    'ymin': bboxes[i][2], 
                    'ymax': bboxes[i][3]
                }
            }
        )

    return utils.json_format(
        code=200, data=data_return,
        message='Success', errors=[]
    )


if __name__ == '__main__':
    app.run(debug=False)