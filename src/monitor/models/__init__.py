from .alex_net import AlexNet
from .vgg import Vgg
from .yolo import Yolo

MODEL_REGISTRY = {
    'alex_net': AlexNet,
    'vgg': Vgg,
    'yolo': Yolo,
}
