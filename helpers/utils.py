import base64
import cv2
import numpy as np
import PIL.Image as Image
import skimage.io
from flask import jsonify


def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def decode(base64_string):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")
    imgdata = base64.b64decode(base64_string)
    img = skimage.io.imread(imgdata, plugin='imageio')
    return img


def convert_image(encoded_image, to_rgb=True):
    img_arr = np.fromstring(encoded_image, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def downsize_img(img, basewidth=480):
    pil_img = Image.fromarray(img)
    wpercent = (basewidth / float(pil_img.size[0]))
    hsize = int((float(pil_img.size[1]) * float(wpercent)))
    rimg = pil_img.resize((basewidth, hsize), Image.ANTIALIAS)
    return np.array(rimg)


def resize_padding(img, desired_size=640):
    """https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    """

    ratio = float(desired_size) / max(img.shape)
    new_size = tuple([int(dim * ratio) for dim in img.shape[:2]])

    # resize img
    rimg = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # make padding
    color = [0, 0, 0]
    rimg = cv2.copyMakeBorder(rimg, top, bottom, left,
                              right, cv2.BORDER_CONSTANT, value=color)

    return rimg


def encode_img(img_path):
    with open(img_path, "rb") as f:
        raw_img = f.read()
    encoded_img = base64.b64encode(raw_img)
    print(encoded_img)


def convert_image_base64(encoded_img, normalize=False):
    decoded_img = base64.b64decode(encoded_img)
    img_arr = np.fromstring(decoded_img, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    if normalize:
        img = img / 255
    return img


def crop_image(img, box):

    sal_crop = img[box[2]:box[3], box[0]:box[1]]
    return sal_crop

def json_format(code=200, message='Default Message!', data=None, errors=None):
    return jsonify({'code': code,
        'data': data,
        'message': message,
        'errors': errors})