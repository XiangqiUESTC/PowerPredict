from .operators.cat import CatPreprocessor
from .operators.softmax import SoftmaxPreprocessor
from .operators.spmm import SpmmPreprocessor

PREPROCESSOR_REGISTRY = {
    "cat": CatPreprocessor,
    "softmax": SoftmaxPreprocessor,
    "spmm": SpmmPreprocessor,
}