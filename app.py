from base64 import b64decode
from io import BytesIO

from flask import Flask, request
from PIL import Image, ImageFile

from cv import detect
from entities.frame import FrameSchema

app = Flask(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


@app.route('/cv', methods=['post'])
def cv():
    schema = FrameSchema()
    data = schema.load(request.get_json())

    image_byte_arr = b64decode(data['image'])
    image_file = BytesIO(image_byte_arr)
    image = Image.open(image_file)

    return detect(image)
