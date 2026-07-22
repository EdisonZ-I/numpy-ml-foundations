# P12: two-layer network backpropagation
import numpy as np

def relu(values):
    result = values.copy()
    result[result < 0] =0
    return result

def softmax(logits):
    process = logits.copy()
    process = process - process.max(axis = 1, keepdims = True)
    process = np.exp(process)
    return process / process.sum(axis = 1, keepdims = True)

'''def two_layer_forward(features, weight1, bias1, weight2, bias2): # from p11
    hidden_linear = features @ weight1 + bias1
    hidden = relu(hidden_linear)
    logits = hidden @ weight2 + bias2
    print(hidden_linear, hidden, logits)
    return logits, {"features": features, "hidden_linear": hidden_linear, "hidden": hidden}'''

def two_layer_loss_and_gradients(
    features,
    labels,
    weight1,
    bias1,
    weight2,
    bias2,
):
    hidden_linear = features @ weight1 + bias1
    hidden = relu(hidden_linear)
    logits = hidden @ weight2 + bias2
    probability = softmax(logits)
    loss = - np.log(probability[np.arange(labels.shape[0]), labels]).mean()
    accuracy = (logits.argmax(axis=1) == labels).sum() / labels.shape[0]

    logits_gradient = probability.copy()
    logits_gradient[np.arange(labels.shape[0]),labels] -= 1
    logits_gradient /= labels.shape[0]

    Weight2_gradient = hidden.T @ logits_gradient

    bias2_gradient = logits_gradient.sum(axis = 0)

    hidden_gradient = logits_gradient @ weight2.T
    hidden_linear_gradient = hidden_gradient * ( hidden_linear >0 )

    weight1_gradient = features.T @ hidden_linear_gradient
    bias1_gradient = hidden_linear_gradient.sum(axis=0)

    gradient = {"weight1": weight1_gradient, "bias1": bias1_gradient, "weight2": Weight2_gradient, "bias2": bias2_gradient, "accuracy": accuracy}

    return loss, gradient

if __name__ == "__main__":
    from p08_numerical_gradient import numerical_gradient
    from test_helpers import check_allclose, check_equal

    features = np.array([[1.0, 2.0], [-1.0, -2.0]])
    labels = np.array([0, 1])
    weight1 = np.array([[0.2, -0.3], [0.4, 0.1]])
    bias1 = np.array([0.1, 0.2])
    weight2 = np.array([[0.5, -0.2], [-0.1, 0.3]])
    bias2 = np.zeros(2)

    loss, gradients = two_layer_loss_and_gradients(
        features, labels, weight1, bias1, weight2, bias2
    )
    numerical_w1 = numerical_gradient(
        lambda value: two_layer_loss_and_gradients(
            features, labels, value, bias1, weight2, bias2
        )[0],
        weight1,
    )
    numerical_w2 = numerical_gradient(
        lambda value: two_layer_loss_and_gradients(
            features, labels, weight1, bias1, value, bias2
        )[0],
        weight2,
    )

    check_allclose(
        "P12/weight1反向传播匹配数值梯度",
        gradients["weight1"],
        numerical_w1,
        rtol=1e-4,
        atol=1e-5,
    )
    check_allclose(
        "P12/weight2反向传播匹配数值梯度",
        gradients["weight2"],
        numerical_w2,
        rtol=1e-4,
        atol=1e-5,
    )
    check_equal("P12/bias1梯度shape", gradients["bias1"].shape, bias1.shape)
    check_equal("P12/bias2梯度shape", gradients["bias2"].shape, bias2.shape)
