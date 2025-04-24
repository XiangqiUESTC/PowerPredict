from .alex_net import AlexNet
from .vgg import Vgg

MODEL_REGISTRY = {
    'alex_net': AlexNet,
    'vgg': vgg
}
