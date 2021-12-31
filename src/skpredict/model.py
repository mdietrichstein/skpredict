from .svm.model import __save_liblinear_model, __save_libsvm_model


def save_model(classifier, data_path):
    __check_for_sklearn()

    from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    from sklearn.svm import NuSVC

    if isinstance(classifier, LinearSVC):
        __save_liblinear_model(classifier, data_path)
    elif isinstance(classifier, SVC) or isinstance(classifier, NuSVC):
        __save_libsvm_model(classifier, data_path)
    else:
        raise ValueError(f"Unable to export unsupported classifier: '{classifier}'")


def __check_for_sklearn():
    import importlib.util

    if not importlib.util.find_spec("sklearn"):
        raise UserWarning(
            "This feature requires sklearn. Please re-install this package with 'pip install skpredict[scikit-learn]'"
        )
