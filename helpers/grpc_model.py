import copy
import cv2

import numpy as np
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import helpers.configs as config

class MaskClassifier(object):
    """docstring for ClassName"""

    def __init__(
        self,
        host=config.HOST,
        port=config.PORT,
        model_name="mask",
        model_signature="mask",
        input_image="input_image",
        y_pred="y_pred",
    ):
        self.host = host
        self.port = port

        self.channel = grpc.insecure_channel("{}:{}".format(self.host, self.port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.input_image = input_image
        self.y_pred = y_pred

        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = model_signature

    def process(self, img):
        img = cv2.resize(img, (150, 150))
        img = img / 255
        return np.expand_dims(img, axis=0)

    def labelFromScore(self, score):
        if score > config.CONFIDENCE_THRESHOLD:
            label = 'Mask'
        else:
            label = 'No mask'
        return label

    def predict(self, img):
        assert img.ndim == 3
        img = self.process(img)

        self.request.inputs[self.input_image].CopyFrom(
            tf.contrib.util.make_tensor_proto(img, dtype=tf.float32, shape=img.shape)
        )

        result = self.stub.Predict(self.request, 10.0)

        y_pred = tf.contrib.util.make_ndarray(result.outputs[self.y_pred])
        score = y_pred[0][0]
        label = self.labelFromScore(score)

        return (label, score)

class Detector(object):
    def __init__(
        self,
        host=config.HOST,
        port=config.PORT,
        model_name="face_detect",
        model_signature="face_detect",
        input_image="input_image",
        y_pred="detect",
    ):
        self.host = host
        self.port = port

        self.channel = grpc.insecure_channel("{}:{}".format(self.host, self.port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.input_image = input_image
        self.y_pred = y_pred

        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_name
        self.request.model_spec.signature_name = model_signature

    def process(self, img):
        img = cv2.resize(img, (300, 300))
        return np.expand_dims(img, axis=0)

    def predict(self, img):
        assert img.ndim == 3
        img_visual = copy.deepcopy(img)
        img_crop = copy.deepcopy(img)
        img_height, img_width, _ = img.shape
        img = self.process(img)
        self.request.inputs[self.input_image].CopyFrom(
            tf.contrib.util.make_tensor_proto(img, dtype=tf.float32, shape=img.shape)
        )

        result = self.stub.Predict(self.request, 10.0)

        y_pred = tf.contrib.util.make_ndarray(result.outputs[self.y_pred])

        confidence_threshold = 0.5
        y_pred_thresh = [
            y_pred[k][y_pred[k, :, 1] > confidence_threshold]
            for k in range(y_pred.shape[0])
        ]
        bbox = list()
        face = list()
        for box in y_pred_thresh[0]:
            if box[2] < 0:
                box[2] = 0
            if box[3] < 0:
                box[3] = 0
            xmin = int(box[2] * img_width / 300)
            ymin = int(box[3] * img_height / 300)
            xmax = int(box[4] * img_width / 300)
            ymax = int(box[5] * img_height / 300)
            cv2.rectangle(img_visual, (xmin, ymin), (xmax, ymax), (0, 0, 220), 3)
            face.append(img_crop[ymin:ymax, xmin:xmax])
            bbox.append([xmin, xmax, ymin, ymax])
        return (img_visual, face)