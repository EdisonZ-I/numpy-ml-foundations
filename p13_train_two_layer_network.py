# P13: train a nonlinear two-layer network
import numpy as np
from p12_two_layer_gradients import two_layer_loss_and_gradients#(features, labels, weight1, bias1, weight2, bias2) -> loss, {gradient}

def train_two_layer_network(
    features,
    labels,
    hidden_size,
    num_classes,
    epochs,
    learning_rate,
    seed,
):
    #init
    rng = np.random.default_rng(seed)
    weight1 = rng.normal(loc = 0, scale = 0.5, size=(features.shape[1],hidden_size))
    bias1 = rng.normal(loc = 0, scale = 0.5, size=(hidden_size, ))

    weight2 = rng.normal(loc = 0, scale = 0.5, size=(hidden_size, num_classes))
    bias2 = rng.normal(loc = 0, scale = 0.5, size=(num_classes, ))

    history =[]

    for epoch in range(epochs):
        loss, gradients = two_layer_loss_and_gradients(features, labels, weight1, bias1, weight2, bias2)
        weight1 -= gradients["weight1"] * learning_rate
        bias1 -= gradients["bias1"] * learning_rate
        weight2 -= gradients["weight2"] * learning_rate
        bias2 -= gradients["bias2"] * learning_rate
        history.append({"epoch": epoch+1, "loss": loss, "accuracy": gradients["accuracy"]})
        print(history[-1])
    parameter = {"weight1": weight1, "bias1": bias1, "weight2": weight2, "bias2": bias2}

    return parameter, history

if __name__ == "__main__":
    from test_helpers import check_equal, check_true
    rng = np.random.default_rng(0)
    features = rng.normal(loc = 2, scale = 5, size = (10, 5))
    labels = rng.integers(0 ,5 , size = 10)

    parameters, history = train_two_layer_network(
        features,
        labels,
        hidden_size=8,
        num_classes=5,
        epochs=1000,
        learning_rate=0.1,
        seed=0,
    )

    check_equal("P13/weight1 shape", parameters["weight1"].shape, (2, 8))
    check_equal("P13/weight2 shape", parameters["weight2"].shape, (8, 2))
    check_equal("P13/每轮都有日志", len(history), 2000)
    check_true(
        "P13/非线性网络训练后损失下降",
        history[-1]["loss"] < history[0]["loss"],
        actual=(history[0]["loss"], history[-1]["loss"]),
        expected="最后loss < 第一轮loss",
    )
    check_equal("P13/最终训练准确率", history[-1]["accuracy"], 1.0) 
        