# P09: analytical gradients for a linear classifier
import numpy as np

def linear_loss_and_gradients(features, labels, weights, bias):
    logits = features @ weights + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    loss = -( (np.log(probabilities[range(labels.shape[0]), labels])).mean() )
    logits_gradient = probabilities.copy()
    logits_gradient[range(labels.shape[0]), labels] -= 1
    logits_gradient /= features.shape[0]

    weights_gradient = features.T @ logits_gradient
    bias_gradient = logits_gradient.sum(axis=0)
    return (loss, weights_gradient, bias_gradient)

    


if __name__ =="__main__":
    from p08_numerical_gradient import numerical_gradient
    from test_helpers import check_allclose

    features = np.array([[1.0, 2.0], [-1.0, 1.0]])
    labels = np.array([0, 1])
    weights = np.array([[0.1, -0.2], [0.3, 0.2]])
    bias = np.array([0.05, -0.05])

    loss, weight_gradient, bias_gradient = linear_loss_and_gradients(
        features, labels, weights, bias
    )
    numerical_w = numerical_gradient(
        lambda value: linear_loss_and_gradients(features, labels, value, bias)[0],
        weights,
    )
    numerical_b = numerical_gradient(
        lambda value: linear_loss_and_gradients(features, labels, weights, value)[0],
        bias,
    )

    check_allclose(
        "P09/weight解析梯度匹配数值梯度",
        weight_gradient,
        numerical_w,
        rtol=1e-4,
        atol=1e-5,
    )
    check_allclose(
        "P09/bias解析梯度匹配数值梯度",
        bias_gradient,
        numerical_b,
        rtol=1e-4,
        atol=1e-5,
    )