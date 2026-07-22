# P05: one-hot encoding and classification metrics
import numpy as np

def one_hot(labels, num_classes):
    if labels[labels >= num_classes].size + labels[labels<0].size > 0 :
        return None
    result = np.zeros((labels.shape[0], num_classes))
    i = range(labels.shape[0])
    result[i, labels[i]] = 1
    return result

def evaluate_logits(logits, labels):
    if logits.shape[0] != labels.shape[0]:
        return None
    result = {"predictions": None, "accuracy": None, "confusion_matrix": None}
    prediction = result["predictions"] = logits.argmax(axis = 1)
    result["accuracy"] = (labels == prediction).sum() / len(labels)
    confusion_matrix = np.zeros((labels.max()+1, labels.max()+1))
    #confusion_matrix[labels, prediction] += 1
    num_classes = logits.shape[1]
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(confusion_matrix, (labels, prediction), 1)
    
    result["confusion_matrix"] = confusion_matrix
    return result

if __name__ =="__main__":
    from test_helpers import check_array_equal, check_equal

    labels = np.array([2, 0, 1])
    check_array_equal(
        "P05/one-hot编码",
        one_hot(labels, 3),
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    )

    logits = np.array([[0.1, 0.9], [3.0, 1.0], [2.0, 4.0]])
    result = evaluate_logits(logits, np.array([1, 1, 0]))
    check_array_equal("P05/logits转预测类别", result["predictions"], np.array([1, 0, 1]))
    check_equal("P05/分类准确率", result["accuracy"], 1 / 3)
    check_array_equal(
        "P05/混淆矩阵",
        result["confusion_matrix"],
        np.array([[0, 1], [1, 1]]),
    )


