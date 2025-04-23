from .avg_pooling import AdaptiveAvgPool2D
from .conv import Conv2D
from .elu import ELU
from .gelu import GELU
from .linear_layer import LinearLayer
from .max_pooling import MaxPool2D
from .relu import ReLU
from .silu import SiLU
from .leaky_relu import Leaky_ReLu
from .mat import Mat

from .spmm import Spmm
from .flatten import Flatten
from .cat import Cat
from .layer_norm import LayerNorm
from .embedding import Embedding
from .positional_encoding import PositionalEncoding
from .roi_align import RoIAlign
from .nms import NMS
from .add import Add
from .softmax import Softmax
from .lstm import LSTM

"""
    在此处注册所有算子类
"""
OPERATOR_REGISTRY = {
    "avg_pooling": AdaptiveAvgPool2D,
    "conv": Conv2D,
    "elu": ELU,
    "gelu": GELU,
    "linear_layer": LinearLayer,
    "max_pooling": MaxPool2D,
    "relu": ReLU,
    "silu": SiLU,
    "leaky_relu": Leaky_ReLu,
    "mat": Mat,

    "spmm": Spmm,
    "flatten": Flatten,
    "cat": Cat,
    "lay_norm": LayerNorm,
    "embedding": Embedding,
    "positional_encoding": PositionalEncoding,
    "roi_align": RoIAlign,
    "nms": NMS,
    "add": Add,
    "softmax": Softmax,
    "lstm": LSTM,
}
