# P11: two-layer ReLU network forward pass
import numpy as np

def relu(values):
    result = values.copy()
    result[result < 0] =0
    return result

def two_layer_forward(features, weight1, bias1, weight2, bias2):
    hidden_linear = features @ weight1 + bias1
    hidden = relu(hidden_linear)
    logits = hidden @ weight2 + bias2
    print(hidden_linear, hidden, logits)
    return logits, {"features": features, "hidden_linear": hidden_linear, "hidden": hidden}

if __name__ == "__main__":
    from test_helpers import check_allclose
    features = np.array([[2.0, 1.0]])
    weight1 = np.array([[1.0, -1.0], [0.5, 1.0]])
    bias1 = np.zeros(2)
    weight2 = np.array([[1.0, -1.0], [2.0, 0.5]])
    bias2 = np.array([0.1, 0.2])

    logits, cache = two_layer_forward(features, weight1, bias1, weight2, bias2)
    check_allclose("P11/两层网络logits", logits, np.array([[2.6, -2.3]]))
    check_allclose(
        "P11/第一层线性输出",
        cache["hidden_linear"],
        np.array([[2.5, -1.0]]),
    )
    check_allclose("P11/ReLU隐藏层输出", cache["hidden"], np.array([[2.5, 0.0]]))
    