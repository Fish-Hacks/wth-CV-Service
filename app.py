from base64 import b64decode
from io import BytesIO

from flask import Flask, request
from PIL import Image, ImageFile

from cv import detect, cuda_init
from entities.frame import FrameSchema

app = Flask(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Run Inital CUDA Check
cuda_init()


@app.route('/cv', methods=['post'])
def cv():
    if 'application/octet-stream' != request.headers['Content-Type']:
        return None

    uuid = request.args['uuid']  # get uuid from the url param.

    image_file = BytesIO(request.data)  # get image from the body
    image = Image.open(image_file)

    result = detect(image)

    return result


@app.route('/cv/base64', methods=['post'])
def cv_base64():
    schema = FrameSchema()
    json = request.get_json()
    data = schema.load(json)

    image_byte_arr = b64decode(data['image'])
    image_file = BytesIO(image_byte_arr)
    image = Image.open(image_file)

    result = detect(image)
    return result
