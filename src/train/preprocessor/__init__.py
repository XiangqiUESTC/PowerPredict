from .raw import Raw
from .operators.cat import CatPreprocessor
from .operators.softmax import SoftmaxPreprocessor
from .operators.spmm import SpmmPreprocessor

PREPROCESSOR_REGISTRY = {
    "raw": Raw,

    "cat": CatPreprocessor,
    "softmax": SoftmaxPreprocessor,
    "spmm": SpmmPreprocessor,
}