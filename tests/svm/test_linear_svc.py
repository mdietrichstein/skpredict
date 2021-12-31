import numpy as np
from sklearn.svm import LinearSVC

from skpredict import liblinear_predict
from skpredict import liblinear_predict_from_model_data
from skpredict import save_model, read_liblinear_model


def test_linear_svc_binary(binary_dataset):
    (X_train, X_test, y_train) = binary_dataset

    classifier = LinearSVC()
    classifier.fit(X_train, y_train)

    samples = X_test.values

    sklearn_predictions = classifier.predict(samples)

    predictions = list(
        [
            liblinear_predict(
                sample,
                classifier.classes_,
                classifier.intercept_,
                classifier.coef_,
                classifier.multi_class,
            )
            for sample in samples
        ]
    )

    np.testing.assert_array_equal(sklearn_predictions, predictions)


def test_linear_svc_multiclass(multiclass_dataset):
    (X_train, X_test, y_train) = multiclass_dataset

    classifier = LinearSVC()
    classifier.fit(X_train, y_train)

    samples = X_test.values[:1]

    sklearn_predictions = classifier.predict(samples)

    predictions = list(
        [
            liblinear_predict(
                sample,
                classifier.classes_,
                classifier.intercept_,
                classifier.coef_,
                classifier.multi_class,
            )
            for sample in samples
        ]
    )

    np.testing.assert_array_equal(sklearn_predictions, predictions)


def test_linear_svc_binary_from_config(binary_dataset, tmp_path):
    (X_train, X_test, y_train) = binary_dataset

    classifier = LinearSVC()
    classifier.fit(X_train, y_train)

    samples = X_test.values

    sklearn_predictions = classifier.predict(samples)

    save_model(classifier, tmp_path / "testmodel.liblinear")
    model = read_liblinear_model(tmp_path / "testmodel.liblinear")

    predictions = list(
        [liblinear_predict_from_model_data(sample, model) for sample in samples]
    )

    np.testing.assert_array_equal(sklearn_predictions, predictions)


def test_linear_svc_multiclass_from_config(multiclass_dataset, tmp_path):
    (X_train, X_test, y_train) = multiclass_dataset

    classifier = LinearSVC()
    classifier.fit(X_train, y_train)

    samples = X_test.values[:1]

    sklearn_predictions = classifier.predict(samples)
    save_model(classifier, tmp_path / "testmodel.liblinear")
    model = read_liblinear_model(tmp_path / "testmodel.liblinear")

    predictions = list(
        [liblinear_predict_from_model_data(sample, model) for sample in samples]
    )

    np.testing.assert_array_equal(sklearn_predictions, predictions)
