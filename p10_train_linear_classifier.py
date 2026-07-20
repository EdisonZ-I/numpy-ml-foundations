# P10: train a softmax linear classifier
import numpy as np
from math import ceil
from p06_shuffled_batches import make_shuffled_batches #(features, labels, batchsize, seed) ->[(features,label)*n]
from p09_linear_gradients import linear_loss_and_gradients #(features, labels, weights, bias) ->(loss, weights_gradient, bias_gradient)


def softmax(logits):
    process = logits.copy()
    process = process - process.max(axis = 1, keepdims = True)
    process = np.exp(process)
    return process / process.sum(axis = 1, keepdims = True)

def loss_and_accuracy(features, labels, weights, bias): #logits.shape[0] == labels.shape[0]
    logits = features @ weights + bias
    prediction = logits.argmax(axis = 1)
    accuracy = (labels == prediction).sum() / labels.shape[0]
    softmax_num = softmax(logits)
    loss = - (np.log(softmax_num[np.arange(labels.shape[0]), labels]).mean())
    return accuracy, loss

def train_linear_classifier(
    features,
    labels,
    num_classes,
    epochs,
    learning_rate,
    batch_size,
    seed,
):
    #init
    weights = np.zeros((features.shape[1], num_classes))
    bias = np.zeros((num_classes, ))
    history = []
    shuffled_batches = make_shuffled_batches(features, labels, batch_size, seed)
    batch_terms = ceil(features.shape[0] / batch_size)

    for epoch in range(epochs):
        for term in range(batch_terms) :#training
           loss, weights_gradient, bias_gradient = linear_loss_and_gradients(shuffled_batches[term][0], shuffled_batches[term][1], weights, bias)
           weights -= weights_gradient * learning_rate
           bias -= bias_gradient * learning_rate
        accuracy, loss = loss_and_accuracy(features, labels, weights, bias)   #record into history
        history.append({"epoch": epoch+1, "loss": loss, "accuracy": accuracy})
        print(history[-1])
    return weights, bias, history


if __name__ == "__main__":
    from test_helpers import check_equal, check_true
    features = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    labels = np.array([0, 0, 1, 1])
    weights, bias, history = train_linear_classifier(
        features,
        labels,
        num_classes=2,
        epochs=40,
        learning_rate=0.5,
        batch_size=2,
        seed=0,
    )

    check_equal("P10/weight shape", weights.shape, (2, 2))
    check_equal("P10/bias shape", bias.shape, (2,))
    check_equal("P10/每轮都有日志", len(history), 40)
    check_true(
        "P10/训练后损失下降",
        history[-1]["loss"] < history[0]["loss"],
        actual=(history[0]["loss"], history[-1]["loss"]),
        expected="最后loss < 第一轮loss",
    )
    check_equal("P10/最终训练准确率", history[-1]["accuracy"], 1.0)