from sys import float_info


def predict(sample, classes, intercepts, coefficients, multi_class_strategy):
    if multi_class_strategy is not None and multi_class_strategy != "ovr":
        raise ValueError(
            f"Multi-class strategy '{multi_class_strategy}' "
            f"not support. Use 'ovr' instead"
        )

    assert len(sample) == len(coefficients[0])

    is_binary = len(classes) == 2

    if is_binary:
        threshold = (
            sum(
                [
                    feature * coefficient
                    for (feature, coefficient) in zip(sample, coefficients[0])
                ]
            )
            + intercepts[0]
        )
        return classes[1] if threshold > 0 else classes[0]

    max_prediction = -float_info.max
    winner_class = -1

    for i, c in enumerate(classes):
        prediction = (
            sum(
                [
                    feature * coefficient
                    for (feature, coefficient) in zip(sample, coefficients[i])
                ]
            )
            + intercepts[i]
        )

        if prediction > max_prediction:
            max_prediction = prediction
            winner_class = c

    return winner_class
