from marshmallow import Schema, fields
from dataclasses import dataclass


@dataclass
class Frame(object):
    def __init__(self, uuid: str, image: str) -> None:
        self.uuid = uuid
        self.image = image


class FrameSchema(Schema):
    uuid = fields.Str()
    image = fields.Str()
