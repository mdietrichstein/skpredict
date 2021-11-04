import numpy as np
from sklearn.svm import LinearSVC

from skpredict import liblinear_predict


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
