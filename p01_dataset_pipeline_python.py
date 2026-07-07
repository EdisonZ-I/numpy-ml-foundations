# P01: paired dataset pipeline in pure Python
from math import ceil

def prepare_dataset(features, labels, test_ratio, batch_size):
    if (not ( 0 <= test_ratio <= 1)) or (batch_size <= 0) or (len(features) != len(labels)): return None
    result ={"train_batches": [], "test": None}
    total_num = len(labels)
    train_num = total_num - int(test_ratio*total_num)
    train_features = features[0:train_num]
    train_labels = labels[0:train_num]
    for i in range( ceil(train_num/batch_size)):
        result["train_batches"].append((train_features[i*batch_size:(i+1)*batch_size], train_labels[i*batch_size:(i+1)*batch_size]))
    result["test"] = (features[train_num:], labels[train_num:])
    return result

if __name__ == "__main__":
    result = prepare_dataset(
        [[1], [2], [3], [4], [5]],
        [10, 20, 30, 40, 50],
        test_ratio=0.4,
        batch_size=2,
    )
    assert result == {
        "train_batches": [([[1], [2]], [10, 20]), ([[3]], [30])],
        "test": ([[4], [5]], [40, 50])
    }

    result = prepare_dataset([[1], [2]], [10, 20], 0, 3)
    assert result == {"train_batches": [([[1], [2]], [10, 20])], "test": ([], [])}
    assert prepare_dataset([[1]], [], 0.2, 2) is None
    