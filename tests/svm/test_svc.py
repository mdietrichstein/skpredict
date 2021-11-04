import numpy as np
import pytest
from sklearn.svm import SVC

from skpredict import libsvm_predict

kernel_values = ["linear", "poly", "rbf", "sigmoid"]

break_ties_values = [False, True]


@pytest.mark.parametrize("kernel", kernel_values)
@pytest.mark.parametrize("break_ties", break_ties_values)
def test_svc_binary(binary_dataset, kernel, break_ties):
    (X_train, X_test, y_train) = binary_dataset

    classifier = SVC(kernel=kernel, break_ties=break_ties)
    classifier.fit(X_train, y_train)

    samples = X_test.values

    sklearn_predictions = classifier.predict(samples)

    print("DFN", classifier.decision_function(samples))

    predictions = list(
        [
            libsvm_predict(
                sample,
                classifier.kernel,
                classifier.classes_,
                classifier.support_vectors_,
                classifier.intercept_,
                classifier.n_support_,
                classifier.dual_coef_,
                break_ties,
                classifier._gamma,
                classifier.coef0,
                classifier.degree,
            )
            for sample in samples
        ]
    )

    np.testing.assert_array_equal(sklearn_predictions, predictions)


@pytest.mark.parametrize("kernel", kernel_values)
@pytest.mark.parametrize("break_ties", break_ties_values)
def test_svc_multiclass_ovr(multiclass_dataset, kernel, break_ties):
    (X_train, X_test, y_train) = multiclass_dataset

    classifier = SVC(kernel=kernel, break_ties=break_ties, decision_function_shape="ovr")
    classifier.fit(X_train, y_train)

    samples = X_test.values[:1]

    sklearn_predictions = classifier.predict(samples)

    predictions = list(
        [
            libsvm_predict(
                sample,
                classifier.kernel,
                classifier.classes_,
                classifier.support_vectors_,
                classifier.intercept_,
                classifier.n_support_,
                classifier.dual_coef_,
                classifier.break_ties,
                classifier._gamma,
                classifier.coef0,
                classifier.degree,
                decision_function_shape=classifier.decision_function_shape,
            )
            for sample in samples
        ]
    )

    np.testing.assert_array_equal(sklearn_predictions, predictions)


@pytest.mark.parametrize("kernel", kernel_values)
def test_svc_multiclass_ovo(multiclass_dataset, kernel):
    (X_train, X_test, y_train) = multiclass_dataset

    classifier = SVC(kernel=kernel, break_ties=False, decision_function_shape="ovo")
    classifier.fit(X_train, y_train)

    samples = X_test.values

    sklearn_predictions = classifier.predict(samples)

    predictions = list(
        [
            libsvm_predict(
                sample,
                classifier.kernel,
                classifier.classes_,
                classifier.support_vectors_,
                classifier.intercept_,
                classifier.n_support_,
                classifier.dual_coef_,
                classifier.break_ties,
                classifier._gamma,
                classifier.coef0,
                classifier.degree,
                decision_function_shape=classifier.decision_function_shape,
            )
            for sample in samples
        ]
    )

    np.testing.assert_array_equal(sklearn_predictions, predictions)
