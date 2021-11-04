from .lib_linear import predict as liblinear_predict
from .lib_svm import predict as libsvm_predict

__version__ = "0.1.0"

__all__ = ["liblinear_predict", "libsvm_predict"]
