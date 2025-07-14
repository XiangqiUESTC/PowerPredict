from .raw import Raw
from .operators.cat import CatPreprocessor
from .operators.softmax import SoftmaxPreprocessor
from .operators.spmm import SpmmPreprocessor
from .operators.lstm import LstmPreprocessor

PREPROCESSOR_REGISTRY = {
    "raw": Raw,
    "lstm": LstmPreprocessor,
    "cat": CatPreprocessor,
    "softmax": SoftmaxPreprocessor,
    "spmm": SpmmPreprocessor,
}