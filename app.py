from flask import Flask, request, jsonify
import numpy as np
import requests
import base64
from helpers import utils
from helpers.grpc_model import MaskClassifier, Detector
from helpers import configs
import json
import time
import cv2
import os

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
    
    (img_visual, faces) = face_dtt.predict(img_arr)

    data_return = list()
    if len(faces) > 0:
        visualized_image_path = os.path.join('media', "image_{}.png".format(time.time()))
        cv2.imwrite(visualized_image_path, img_visual[:,:,::-1])
        data_return.append({'face_detected_image': str(visualized_image_path)})
        message = 'Success'
    else:
        message = 'No face detected'
    for i in range(len(faces)):
        label, score = classifier.predict(faces[i])
        face_image_path = os.path.join('media', "face_{}.png".format(time.time()))
        cv2.imwrite(face_image_path, faces[i][:,:,::-1])
    
        data_return.append(
            {
                'label': str(label),
                'score': float(score),
                'croped_face_image': str(face_image_path),
            }
        )
        

    return utils.json_format(
        code=200, 
        data=data_return,
        message=message,
        errors=[]
    )


if __name__ == '__main__':
    if not os.path.exists('media'):
        os.makedirs('media')
    app.run(debug=False)
    