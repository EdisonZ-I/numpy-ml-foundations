# P03: pixel normalization without data leakage
import numpy as np

def preprocess_pixels(train_features, test_features):
    if train_features.shape[1] != test_features.shape[1]:
        return None
    train_features1 = train_features / 255.0
    test_features1 = test_features / 255.0
    
    mean = train_features1.mean(axis=0)
    scale = np.std(train_features1, axis=0)
    scale [scale == 0] = 1.0
    processed_train = (train_features1 - mean) / scale
    processed_test = (test_features1 - mean) / scale
    return (processed_train, processed_test, mean, scale)

if __name__ =="__main__":
    from test_helpers import check_allclose, check_array_equal

    train = np.array([[0, 255], [255, 255]])
    test = np.array([[127.5, 255]])
    train_before = train.copy()

    processed_train, processed_test, mean, scale = preprocess_pixels(train, test)
    check_allclose(
        "P03/训练集标准化",
        processed_train,
        np.array([[-1.0, 0.0], [1.0, 0.0]]),
    )
    check_allclose("P03/测试集复用训练统计量", processed_test, np.array([[0.0, 0.0]]))
    check_allclose("P03/训练集均值", mean, np.array([0.5, 1.0]))
    check_allclose("P03/零标准差替换为一", scale, np.array([0.5, 1.0]))
    check_array_equal("P03/不修改输入", train, train_before)