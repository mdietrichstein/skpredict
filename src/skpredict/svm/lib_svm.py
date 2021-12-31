import numpy as np


def predict_from_model_data(sample, model_data):
    header = model_data["header"]

    kernel_type = header["kernel_type"]

    if kernel_type == "polynomial":
        kernel = "poly"
    elif kernel_type == "gaussian":
        kernel = "rbf"
    else:
        kernel = kernel_type

    classes = header["label"]
    intercepts = list([-rho for rho in header["rho"]])
    number_of_support_vectors = header["nr_sv"]

    if header.get("break_ties") == 1:
        break_ties = True
    else:
        break_ties = False

    gamma = header.get("gamma")
    independent_term = header.get("coef0")
    degree = header.get("degree")

    if "decision_function_shape" in header:
        decision_function_shape = header["decision_function_shape"]
    else:
        decision_function_shape = "ovr"

    dual_coeffs = model_data["dual_coeffs"]
    support_vectors = model_data["support_vectors"]

    return predict(
        sample,
        kernel,
        classes,
        support_vectors,
        intercepts,
        number_of_support_vectors,
        dual_coeffs,
        break_ties,
        gamma,
        independent_term,
        degree,
        decision_function_shape,
    )


def predict(
    sample,
    kernel,
    classes,
    support_vectors,
    intercepts,
    num_support_vectors_per_class,
    dual_coefficients,
    break_ties=False,
    gamma=None,
    r=None,
    degree=None,
    decision_function_shape="ovr",
):
    if break_ties and decision_function_shape == "ovo":
        raise ValueError(
            "break_ties must be False when " "decision_function_shape is 'ovo'"
        )
    fn_kernel = None

    if kernel == "linear":
        fn_kernel = __kernel_linear
    elif kernel == "poly":
        assert gamma is not None
        fn_kernel = __create_kernel_poly(gamma, r, degree)
    elif kernel == "rbf":
        assert gamma is not None
        fn_kernel = __create_kernel_rbf(gamma)
    elif kernel == "sigmoid":
        assert gamma is not None
        fn_kernel = __create_kernel_sigmoid(gamma, r)
    else:
        raise ValueError(f'Unsupported kernel "{kernel}"')

    num_classes = len(classes)

    support_vector_indices = np.hstack(([0], np.cumsum(num_support_vectors_per_class)))

    convolutions = []

    for i, support_vector in enumerate(support_vectors):
        convolutions.append(fn_kernel(support_vector, sample))

    def dual_coeff_times_conv_for_svm(class_a_index, class_b_index):
        sv_idx_start = support_vector_indices[class_b_index]
        sv_idx_end = support_vector_indices[class_b_index + 1]

        convs = convolutions[sv_idx_start:sv_idx_end]
        offset = -1 if class_a_index > class_b_index else 0
        instance_dual_coefficients = dual_coefficients[
            class_a_index + offset, sv_idx_start:sv_idx_end
        ]

        return sum(
            [
                dual_coeff * conv
                for dual_coeff, conv in zip(instance_dual_coefficients, convs)
            ]
        )

    num_svms = int((num_classes * (num_classes - 1)) / 2)
    confidences = np.zeros((num_svms,))

    svm_idx = 0
    for class_a_idx in range(0, num_classes):
        for class_b_idx in range(class_a_idx + 1, num_classes):
            confidence = (
                dual_coeff_times_conv_for_svm(class_a_idx, class_b_idx)
                + dual_coeff_times_conv_for_svm(class_b_idx, class_a_idx)
                + intercepts[svm_idx]
            )

            confidences[svm_idx] = confidence
            svm_idx += 1

    if decision_function_shape == "ovr":
        predicted_class_idx = __ovr_decision_function(
            confidences, num_classes, break_ties
        )
    elif decision_function_shape == "ovo":
        predicted_class_idx = __ovo_decision_function(confidences, num_classes)
    else:
        raise ValueError(
            f"Unsupported decision_function_shape '{decision_function_shape}'"
        )

    return classes[predicted_class_idx]


def __kernel_linear(support_vector, sample):
    assert len(support_vector) == len(sample)
    return np.dot(support_vector, sample)


def __kernel_poly(support_vector, sample, gamma, r, degree):
    assert len(support_vector) == len(sample)
    return np.power(gamma * np.dot(support_vector, sample) + r, degree)


def __kernel_rbf(support_vector, sample, gamma):
    assert gamma > 0
    assert len(support_vector) == len(sample)

    # https://stackoverflow.com/a/ 27752709
    # https://stackoverflow.com/questions/28503932/calculating-decision-function-of-svm-manually
    # https://stackoverflow.com/questions/21800301/svm-scikit-learn-decision-values-with-rbf-kernel
    return np.exp(
        -gamma
        * np.dot(
            np.subtract(support_vector, sample),
            np.subtract(support_vector, sample),
        )
    )


def __kernel_sigmoid(support_vector, sample, gamma, r):
    assert len(support_vector) == len(sample)
    return np.tanh(gamma * np.dot(support_vector, sample) + r)


def __create_kernel_poly(gamma, r, degree):
    def fn(support_vector, sample):
        return __kernel_poly(support_vector, sample, gamma, r, degree)

    return fn


def __create_kernel_rbf(gamma):
    def fn(support_vector, sample):
        return __kernel_rbf(support_vector, sample, gamma)

    return fn


def __create_kernel_sigmoid(gamma, r):
    def fn(support_vector, sample):
        return __kernel_sigmoid(support_vector, sample, gamma, r)

    return fn


def __ovo_decision_function(confidences, num_classes):
    votes = np.zeros((num_classes,))
    svm_idx = 0
    for class_a_idx in range(0, num_classes):
        for class_b_idx in range(class_a_idx + 1, num_classes):
            predicted_class_idx = class_a_idx if confidences[svm_idx] > 0 else class_b_idx
            votes[predicted_class_idx] += 1
            svm_idx += 1

    return np.argmax(votes)


def __ovr_decision_function(confidences, num_classes, break_ties):
    # https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/utils/multiclass.py#L423
    predictions = np.array(confidences) < 0
    confidences = -confidences

    votes = np.zeros((num_classes,))
    sum_of_confidences = np.zeros((num_classes,))

    svm_idx = 0
    for class_a_idx in range(0, num_classes):
        for class_b_idx in range(class_a_idx + 1, num_classes):
            sum_of_confidences[class_a_idx] -= confidences[svm_idx]
            sum_of_confidences[class_b_idx] += confidences[svm_idx]
            votes[predictions[svm_idx] == 0, class_a_idx] += 1
            votes[predictions[svm_idx] == 1, class_b_idx] += 1
            svm_idx += 1

    confidences = sum_of_confidences / (3 * (np.abs(sum_of_confidences) + 1))
    confidences = votes + confidences

    if break_ties:
        predicted_class_idx = np.argmax(confidences)
    else:
        predicted_class_idx = np.argmax(votes)

    is_binary = num_classes == 2

    if is_binary:
        predicted_class_idx = abs(predicted_class_idx - 1)

    return predicted_class_idx
