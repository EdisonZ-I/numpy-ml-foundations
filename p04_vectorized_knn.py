# P04: vectorized distances and 1-NN
import numpy as np

def pairwise_squared_distances(test_features, train_features):
    if test_features.shape[1] != train_features.shape[1]:
        return None
    i, j = np.indices((test_features.shape[0], test_features.shape[0]))
    diff_square = (train_features[i] - test_features[j]) ** 2
    result = diff_square.sum(axis = 2).T
    return result

def predict_1nn(train_features, train_labels, test_features):
    if test_features.shape != train_features.shape:
        return None
    maxarg = pairwise_squared_distances(test_features, train_features).argmin(axis = 1)
    result = train_labels[maxarg]
    return result

if __name__ == "__main__":
    from test_helpers import check_array_equal

    train_x = np.array([[0, 0], [3, 3]])
    train_y = np.array([0, 1])
    test_x = np.array([[1, 0], [2, 2]])

    check_array_equal(
        "P04/广播计算距离矩阵",
        pairwise_squared_distances(test_x, train_x),
        np.array([[1, 13], [8, 2]]),
    )
    check_array_equal(
        "P04/最近邻预测",
        predict_1nn(train_x, train_y, test_x),
        np.array([0, 1]),
    )