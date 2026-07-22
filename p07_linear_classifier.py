# P07: stable softmax linear classifier
import numpy as np

def softmax(logits):
    safe = logits - logits.max(axis = 1, keepdims = True)
    safe = np.exp(safe)
    safe = safe / safe.sum(axis=1, keepdims = True)
    return safe

def linear_classifier_loss(features, labels, weights, bias):
    logits = features @ weights + bias
    prob = softmax(logits)
    pred = prob.argmax(axis = 1)
    loss_list = -np.log(prob[labels])
    loss = loss_list.mean()
    return (loss, prob, pred)

if __name__ == "__main__":
    from test_helpers import check_allclose, check_array_equal, check_true
    features = np.eye(2)
    labels = np.array([0, 1])
    weights = np.zeros((2, 2))
    bias = np.zeros(2)

    loss, probabilities, predictions = linear_classifier_loss(features, labels, weights, bias)
    check_allclose("P07/零参数时的交叉熵", loss, np.log(2.0))
    check_allclose("P07/零参数时类别等概率", probabilities, np.full((2, 2), 0.5))
    check_array_equal("P07/并列时选择首个类别", predictions, np.array([0, 0]))

    stable = softmax(np.array([[1000.0, 1001.0, 1002.0]]))
    check_true(
        "P07/大logits仍产生有限概率",
        np.all(np.isfinite(stable)),
        actual=stable,
        expected="所有元素均为有限数",
    )
    check_allclose("P07/softmax每行概率和为一", stable.sum(axis=1), np.ones(1))