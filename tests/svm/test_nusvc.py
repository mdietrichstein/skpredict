import numpy as np
import pytest
from sklearn.svm import NuSVC

from skpredict import libsvm_predict, libsvm_predict_from_model_data
from skpredict import save_model, read_libsvm_model

kernel_values = ["linear", "poly", "rbf", "sigmoid"]

break_ties_values = [False, True]

NU = 0.05


@pytest.mark.parametrize("kernel", kernel_values)
@pytest.mark.parametrize("break_ties", break_ties_values)
def test_nusvc_binary(binary_dataset, kernel, break_ties):
    (X_train, X_test, y_train) = binary_dataset

    classifier = NuSVC(kernel=kernel, break_ties=break_ties, nu=NU)
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
def test_nusvc_multiclass_ovr(multiclass_dataset, kernel, break_ties):
    (X_train, X_test, y_train) = multiclass_dataset

    classifier = NuSVC(kernel=kernel, break_ties=break_ties, nu=NU)
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
            )
            for sample in samples
        ]
    )

    np.testing.assert_array_equal(sklearn_predictions, predictions)


@pytest.mark.parametrize("kernel", kernel_values)
def test_nusvc_multiclass_ovo(multiclass_dataset, kernel):
    (X_train, X_test, y_train) = multiclass_dataset

    classifier = NuSVC(
        kernel=kernel, break_ties=False, nu=NU, decision_function_shape="ovo"
    )
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
            )
            for sample in samples
        ]
    )

    np.testing.assert_array_equal(sklearn_predictions, predictions)


@pytest.mark.parametrize("kernel", kernel_values)
@pytest.mark.parametrize("break_ties", break_ties_values)
def test_nusvc_binary_from_config(binary_dataset, kernel, break_ties, tmp_path):
    (X_train, X_test, y_train) = binary_dataset

    classifier = NuSVC(kernel=kernel, break_ties=break_ties, nu=NU)
    classifier.fit(X_train, y_train)

    samples = X_test.values

    sklearn_predictions = classifier.predict(samples)

    save_model(classifier, tmp_path / "model.libsvm")
    model = read_libsvm_model(tmp_path / "model.libsvm")

    predictions = list(
        [libsvm_predict_from_model_data(sample, model) for sample in samples]
    )

    np.testing.assert_array_equal(sklearn_predictions, predictions)


@pytest.mark.parametrize("kernel", kernel_values)
@pytest.mark.parametrize("break_ties", break_ties_values)
def test_nusvc_multiclass_ovr_from_config(
    multiclass_dataset, kernel, break_ties, tmp_path
):
    (X_train, X_test, y_train) = multiclass_dataset

    classifier = NuSVC(kernel=kernel, break_ties=break_ties, nu=NU)
    classifier.fit(X_train, y_train)

    samples = X_test.values

    sklearn_predictions = classifier.predict(samples)

    save_model(classifier, tmp_path / "model.libsvm")
    model = read_libsvm_model(tmp_path / "model.libsvm")

    predictions = list(
        [libsvm_predict_from_model_data(sample, model) for sample in samples]
    )

    np.testing.assert_array_equal(sklearn_predictions, predictions)


@pytest.mark.parametrize("kernel", kernel_values)
def test_nusvc_multiclass_ovo_from_config(multiclass_dataset, kernel, tmp_path):
    (X_train, X_test, y_train) = multiclass_dataset

    classifier = NuSVC(
        kernel=kernel, break_ties=False, nu=NU, decision_function_shape="ovo"
    )
    classifier.fit(X_train, y_train)

    samples = X_test.values

    sklearn_predictions = classifier.predict(samples)

    save_model(classifier, tmp_path / "model.libsvm")
    model = read_libsvm_model(tmp_path / "model.libsvm")

    predictions = list(
        [libsvm_predict_from_model_data(sample, model) for sample in samples]
    )

    np.testing.assert_array_equal(sklearn_predictions, predictions)
