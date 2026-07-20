# P06: reproducible shuffled mini-batches
import numpy as np
from math import ceil

def make_shuffled_batches(features, labels, batch_size, seed):
    if features.shape[0] != labels.shape[0] or batch_size <=0:
        return None
    features_shuffled = np.random.default_rng(seed).permutation(features)
    labels_shuffled = np.random.default_rng(seed).permutation(labels)
    result = []
    for i in range(ceil(labels.shape[0]/batch_size)):
        result.append((features_shuffled[i*batch_size:(i+1)*batch_size,:],labels_shuffled[i*batch_size:(i+1)*batch_size]))
    return result

if __name__ == "__main__":
    from test_helpers import check_array_equal, check_equal
    features = np.array([[10], [20], [30], [40], [50]])
    labels = np.array([1, 2, 3, 4, 5])
    batches_a = make_shuffled_batches(features, labels, 2, seed=42)
    batches_b = make_shuffled_batches(features, labels, 2, seed=42)

    x_a = np.concatenate([x for x, _ in batches_a])
    y_a = np.concatenate([y for _, y in batches_a])
    x_b = np.concatenate([x for x, _ in batches_b])
    y_b = np.concatenate([y for _, y in batches_b])

    check_array_equal("P06/相同seed产生相同特征顺序", x_a, x_b)
    check_array_equal("P06/相同seed产生相同标签顺序", y_a, y_b)
    check_array_equal("P06/打乱后特征标签仍配对", x_a[:, 0] / 10, y_a)
    check_equal("P06/batch大小包含末尾余数", [len(x) for x, _ in batches_a], [2, 2, 1])
