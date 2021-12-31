def read_liblinear_model(data_path):
    return __read_liblinear_model(data_path)


def read_libsvm_model(data_path):
    return __read_libsvm_model(data_path)


# https://github.com/cjlin1/liblinear/blob/60f1adf6f35d6f3e031c334b33dfe8399d6f8a9d/linear.cpp#L3396
def __save_liblinear_model(classifier, data_path):
    import locale

    current_locale = locale.setlocale(locale.LC_ALL, None)

    try:
        locale.setlocale(locale.LC_ALL, "C")

        with open(data_path, "w") as f:
            num_classes = len(classifier.classes_)
            num_features = len(classifier.coef_[0])

            if num_classes == 2:
                num_weights = 1
            else:
                num_weights = num_classes

            # f.write(f"solver_type {solver_type}\n") # we do not have that info when using scikit-learn
            f.write(f"nr_class {num_classes:d}\n")

            f.write("label")
            for c in classifier.classes_:
                f.write(f" {c:d}")
            f.write("\n")

            f.write(f"nr_feature {num_features:d}\n")

            # Custom property required by scikit-learn. Determines how multi-class predictions are performed
            f.write(f"multi_class_strategy {classifier.multi_class}\n")

            print(f"C1 {classifier.coef_.shape}, {len(classifier.classes_)}")
            f.write("w\n")
            for i in range(0, num_features):
                for j in range(0, num_weights):
                    f.write(f"{classifier.coef_[j, i]:.17g} ")
                f.write("\n")

            for j in range(0, num_weights):
                f.write(f"{classifier.intercept_[j]:.17g} ")

            f.write("\n")
    finally:
        locale.setlocale(locale.LC_ALL, current_locale)


def __read_liblinear_model_header(f):
    model_header = {}

    while True:
        line = f.readline()

        if not line:
            break

        parts = line.split(" ")

        key = parts[0].strip()

        if key in ["nr_class", "nr_feature"]:
            model_header[key] = int(parts[1].strip())
        elif key == "label":
            model_header[key] = list([int(p.strip()) for p in parts[1:]])
        elif key == "multi_class_strategy":
            model_header[key] = parts[1].strip()
        elif key == "w":
            # weights (w) should always appear after all header fields
            break

    return model_header


def __read_liblinear_model(data_path):
    import locale
    import numpy as np

    current_locale = locale.setlocale(locale.LC_ALL, None)

    try:
        locale.setlocale(locale.LC_ALL, "C")
        with open(data_path, "r") as f:
            model_header = __read_liblinear_model_header(f)

            num_classes = model_header["nr_class"]
            num_features = model_header["nr_feature"]

            if num_classes == 2:
                coefficients = np.zeros((1, num_features))
            else:
                coefficients = np.zeros((num_classes, num_features))

            intercepts = np.zeros((num_classes,))

            processed_weights = 0

            line = f.readline()
            while line:
                parts = line.strip().split(" ")

                if processed_weights < num_features:
                    print(coefficients.shape, len(parts), num_classes)
                    coefficients[:, processed_weights] = [float(w) for w in parts]
                elif processed_weights == num_features:
                    intercepts[:] = [float(w) for w in parts]
                else:
                    raise ValueError("More entries than required weights")

                processed_weights += 1
                line = f.readline()

            return {
                "header": model_header,
                "coefficients": coefficients,
                "intercepts": intercepts,
            }

    finally:
        locale.setlocale(locale.LC_ALL, current_locale)


__KERNEL_TYPE_LINEAR = "linear"
__KERNEL_TYPE_POLY = "polynomial"
__KERNEL_TYPE_RBF = "gaussian"
__KERNEL_TYPE_SIGMOID = "sigmoid"


# https://github.com/cjlin1/libsvm/blob/557d85749aaf0ca83fd229af0f00e4f4cb7be85c/svm.cpp#L2647
def __save_libsvm_model(classifier, data_path):
    import locale
    from sklearn.svm import SVC
    from sklearn.svm import NuSVC

    if isinstance(classifier, SVC):
        svm_type = "c_svc"
    elif isinstance(classifier, NuSVC):
        svm_type = "nu_svc"
    else:
        raise ValueError(f"Unable to determine svm_type for classifier: '{classifier}'")

    kernel = classifier.kernel

    if kernel == "linear":
        kernel_type = __KERNEL_TYPE_LINEAR
    elif kernel == "poly":
        kernel_type = __KERNEL_TYPE_POLY
    elif kernel == "rbf":
        kernel_type = __KERNEL_TYPE_RBF
    elif kernel == "sigmoid":
        kernel_type = __KERNEL_TYPE_SIGMOID
    else:
        raise ValueError(f'Unsupported kernel "{kernel}"')

    current_locale = locale.setlocale(locale.LC_ALL, None)

    try:
        locale.setlocale(locale.LC_ALL, "C")

        with open(data_path, "w") as f:
            f.write(f"svm_type {svm_type}\n")
            f.write(f"kernel_type {kernel_type}\n")

            # "break_ties" and "decision_function_shape" are custom properties which influence how multi-class
            # classification is performed in scikit-learn..
            f.write(f"break_ties {1 if classifier.break_ties else 0:d}\n")
            f.write(f"decision_function_shape {classifier.decision_function_shape}\n")

            # "nr_features" is a custom property which specifies the number of features.
            # We need this to obtain the correct support vector shape when reading the model.
            # The libsvm model file format does not specify this field.
            f.write(f"nr_features {int(classifier.support_vectors_.shape[1])}\n")

            if kernel_type == __KERNEL_TYPE_POLY:
                f.write(f"degree {classifier.degree:d}\n")

            if kernel_type in [
                __KERNEL_TYPE_POLY,
                __KERNEL_TYPE_RBF,
                __KERNEL_TYPE_SIGMOID,
            ]:
                f.write(f"gamma {classifier._gamma:.17g}\n")

            if kernel_type in [__KERNEL_TYPE_POLY, __KERNEL_TYPE_SIGMOID]:
                f.write(f"coef0 {classifier.coef0:.17g}\n")

            num_classes = len(classifier.classes_)
            num_svms = int((num_classes * (num_classes - 1)) / 2)

            f.write(f"nr_class {num_classes:d}\n")
            f.write(f"total_sv {len(classifier.support_vectors_):d}\n")

            f.write("nr_sv")
            for n_support in classifier.n_support_:
                f.write(f" {n_support:d}")
            f.write("\n")

            f.write("label")
            for c in classifier.classes_:
                f.write(f" {c:d}")
            f.write("\n")

            # rho should be the negative intercept. I haven't found reliable information on what rho actually means,
            # so this might be wrong
            assert len(classifier.intercept_) == num_svms
            f.write("rho")
            for intercept in classifier.intercept_:
                f.write(f" {-intercept:.17g}")
            f.write("\n")

            f.write("SV\n")
            for i, sv in enumerate(classifier.support_vectors_):
                for j in range(0, num_classes - 1):
                    f.write(f"{classifier.dual_coef_[j][i]:.17g} ")

                for index, value in enumerate(sv):
                    if value != 0:
                        f.write(f"{(index + 1):d}:{value:.8g} ")
                f.write("\n")
    finally:
        locale.setlocale(locale.LC_ALL, current_locale)


def __read_libsvm_model_header(f):
    model_header = {}

    while True:
        line = f.readline()

        if not line:
            break

        parts = line.split(" ")

        key = parts[0].strip()

        if key in ["svm_type", "kernel_type", "decision_function_shape"]:
            model_header[key] = parts[1].strip()
        elif key in ["break_ties", "degree", "nr_class", "total_sv", "nr_features"]:
            model_header[key] = int(parts[1].strip())
        elif key in ["gamma", "coef0"]:
            model_header[key] = float(parts[1].strip())
        elif key in ["nr_sv", "label"]:
            model_header[key] = list([int(p.strip()) for p in parts[1:]])
        elif key == "rho":
            model_header[key] = list([float(p.strip()) for p in parts[1:]])
        elif key == "SV":
            # SV should always appear after all header fields
            break

    return model_header


def __read_libsvm_model(data_path):
    import locale
    import numpy as np

    current_locale = locale.setlocale(locale.LC_ALL, None)

    try:
        locale.setlocale(locale.LC_ALL, "C")
        with open(data_path, "r") as f:
            model_header = __read_libsvm_model_header(f)

            num_classes = model_header["nr_class"]
            num_features = model_header["nr_features"]
            num_sv = model_header["total_sv"]

            dual_coeffs = np.zeros((num_classes - 1, num_sv))
            support_vectors = np.zeros((num_sv, num_features))

            line = f.readline()

            processed_sv = 0
            while line:
                parts = line.strip().split(" ")

                for i, part in enumerate(parts):
                    if i < num_classes - 1:
                        dual_coeffs[i, processed_sv] = float(part)
                    else:
                        sv_parts = part.split(":")
                        index, value = int(sv_parts[0]), float(sv_parts[1])
                        support_vectors[processed_sv, index - 1] = value

                processed_sv += 1

                line = f.readline()

            assert processed_sv == num_sv

            return {
                "header": model_header,
                "dual_coeffs": dual_coeffs,
                "support_vectors": support_vectors,
            }

    finally:
        locale.setlocale(locale.LC_ALL, current_locale)
