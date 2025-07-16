from .raw import Raw
from .operators.cat import CatPreprocessor
from .operators.softmax import SoftmaxPreprocessor
from .operators.spmm import SpmmPreprocessor
from .operators.lstm import LstmPreprocessor
from .operators.mat import MatPreprocessor
from .operators.add import AddPreprocessor
from .operators.max_pooling import  MaxPoolingPreprocessor
from .operators.conv import ConvPreprocessor
PREPROCESSOR_REGISTRY = {
    "raw": Raw,
    "lstm": LstmPreprocessor,
    "cat": CatPreprocessor,
    "softmax": SoftmaxPreprocessor,
    "spmm": SpmmPreprocessor,
    "mat":MatPreprocessor,
    "conv": ConvPreprocessor,
    "add": AddPreprocessor,
    "max_pooling": MaxPoolingPreprocessor,

}