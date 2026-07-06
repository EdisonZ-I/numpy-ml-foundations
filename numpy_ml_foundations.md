# NumPy and ML Foundations Practice

目标：从纯 Python 平稳过渡到 NumPy，掌握机器学习中常见的数据表示、向量化计算、分类评估和梯度下降，为从零实现 MNIST 分类器做准备。

建议节奏：

- 每次只做 1 到 3 题。
- 每题写在指定的 `.py` 文件中。
- 先独立完成，再运行基础断言。
- 基础断言通过只代表初步正确，批改时还会补充边界测试。
- 除非题目明确允许，进入 NumPy 阶段后尽量不要用 Python 循环代替向量化操作。

环境准备：

```powershell
python -m pip install -r requirements.txt
python -c "import numpy as np; print(np.__version__)"
```

通用约定：

- NumPy 统一写作 `import numpy as np`。
- 浮点数组使用 `np.testing.assert_allclose` 测试。
- 整数数组使用 `np.testing.assert_array_equal` 测试。
- 不要修改输入数组，除非题目明确要求原地修改。
- shape 不匹配、长度不匹配等情况，按每题要求处理。
- 函数名和参数名尽量保持题目给出的形式。

推荐顺序：严格按 P01-P24 完成。P01-P04 是纯 Python 过渡题，P05 开始正式使用 NumPy。

## 第一阶段：纯 Python 的 ML 数据处理过渡

### P01：混淆矩阵

文件：`p01_confusion_matrix_python.py`

实现：

```python
def confusion_matrix(labels, predictions, num_classes):
    ...
```

要求：

- 返回 `num_classes × num_classes` 的二维列表。
- 行表示真实类别，列表示预测类别。
- 对每个样本执行 `matrix[真实类别][预测类别] += 1`。
- `labels` 与 `predictions` 长度不同或 `num_classes <= 0` 时返回 `None`。
- 可以假设所有标签都在 `0` 到 `num_classes - 1` 范围内。

自测：

```python
assert confusion_matrix([0, 1, 1, 2], [0, 2, 1, 2], 3) == [
    [1, 0, 0],
    [0, 1, 1],
    [0, 0, 1],
]
assert confusion_matrix([], [], 2) == [[0, 0], [0, 0]]
assert confusion_matrix([0], [0, 1], 2) is None
```

### P02：同步切分特征和标签

文件：`p02_split_xy_python.py`

实现：

```python
def train_test_split_xy(features, labels, test_ratio):
    ...
```

要求：

- 不打乱原顺序。
- 测试集数量使用 `int(len(features) * test_ratio)`。
- 测试集取最后一段。
- 返回 `(x_train, x_test, y_train, y_test)`。
- 特征与标签长度不同，或 `test_ratio` 不在 `[0, 1]` 内时返回 `None`。
- 特别注意测试集数量为 `0` 时的切片行为。

自测：

```python
result = train_test_split_xy([[1], [2], [3], [4]], [0, 0, 1, 1], 0.25)
assert result == ([[1], [2], [3]], [[4]], [0, 0, 1], [1])

result = train_test_split_xy([[1], [2]], [0, 1], 0)
assert result == ([[1], [2]], [], [0, 1], [])
assert train_test_split_xy([[1]], [], 0.2) is None
```

### P03：同步制作 mini-batch

文件：`p03_batches_xy_python.py`

实现：

```python
def make_batches_xy(features, labels, batch_size):
    ...
```

要求：

- 返回由 `(x_batch, y_batch)` 组成的列表。
- 最后一个 batch 可以不足 `batch_size`。
- 必须保持每条特征与对应标签的配对关系。
- 长度不同或 `batch_size <= 0` 时返回 `None`。

自测：

```python
assert make_batches_xy([[1], [2], [3]], [10, 20, 30], 2) == [
    ([[1], [2]], [10, 20]),
    ([[3]], [30]),
]
assert make_batches_xy([], [], 2) == []
assert make_batches_xy([[1]], [], 2) is None
```

### P04：纯 Python 的 1-NN 分类器

文件：`p04_knn_python.py`

实现：

```python
def euclidean_distance(point_a, point_b):
    ...

def predict_1nn(train_features, train_labels, test_features):
    ...
```

要求：

- `euclidean_distance` 计算两个等长向量的欧氏距离；长度不同返回 `None`。
- `predict_1nn` 为每个测试样本寻找距离最近的训练样本，并返回其标签。
- 距离相同时选择训练集中先出现的样本。
- 训练特征与标签长度不同，或训练集为空时返回 `None`。
- 本题先用列表和循环完成。

自测：

```python
assert euclidean_distance([0, 0], [3, 4]) == 5
assert euclidean_distance([1], [1, 2]) is None

train_x = [[0, 0], [3, 3]]
train_y = [0, 1]
assert predict_1nn(train_x, train_y, [[1, 0], [2.5, 2.5]]) == [0, 1]
assert predict_1nn([], [], [[1, 1]]) is None
```

## 第二阶段：NumPy 数组、shape 与索引

### P05：数组基本属性

文件：`p05_array_info.py`

实现：

```python
import numpy as np

def array_info(values):
    ...
```

要求：

- 用 `np.asarray` 将输入转换成数组。
- 返回包含 `shape`、`ndim`、`size` 的字典。
- `shape` 保持 NumPy 的元组形式。

自测：

```python
assert array_info([[1, 2, 3], [4, 5, 6]]) == {
    "shape": (2, 3),
    "ndim": 2,
    "size": 6,
}
assert array_info(5) == {"shape": (), "ndim": 0, "size": 1}
```

### P06：索引与切片

文件：`p06_indexing_slicing.py`

实现：

```python
import numpy as np

def select_parts(matrix):
    ...
```

要求：

- 输入一个至少为 `2 × 2` 的二维数组。
- 返回 `(第二行, 最后一列, 左上角2×2区域)`。
- 返回值都应为 NumPy 数组。
- 使用索引和切片，不使用循环。

自测：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
second_row, last_column, top_left = select_parts(matrix)
np.testing.assert_array_equal(second_row, np.array([4, 5, 6]))
np.testing.assert_array_equal(last_column, np.array([3, 6, 9]))
np.testing.assert_array_equal(top_left, np.array([[1, 2], [4, 5]]))
```

### P07：reshape 与 flatten

文件：`p07_reshape.py`

实现：

```python
import numpy as np

def reshape_flat_images(flat_images, height, width):
    ...

def flatten_images(images):
    ...
```

要求：

- `flat_images` 的 shape 为 `(样本数, height * width)`。
- `reshape_flat_images` 返回 `(样本数, height, width)`。
- 宽度不匹配时返回 `None`。
- `flatten_images` 把 `(样本数, height, width)` 变成 `(样本数, height * width)`。
- 不使用循环。

自测：

```python
flat = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
images = reshape_flat_images(flat, 2, 2)
np.testing.assert_array_equal(
    images,
    np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
)
np.testing.assert_array_equal(flatten_images(images), flat)
assert reshape_flat_images(flat, 3, 2) is None
```

### P08：理解 axis

文件：`p08_axis_statistics.py`

实现：

```python
import numpy as np

def array_statistics(matrix):
    ...
```

要求：

- 返回 `(每行之和, 每列平均值, 每行最大值的索引)`。
- 分别使用 `sum`、`mean`、`argmax` 的 `axis` 参数。
- 不使用循环。

自测：

```python
matrix = np.array([[1, 5, 3], [4, 2, 9]])
row_sums, column_means, row_argmax = array_statistics(matrix)
np.testing.assert_array_equal(row_sums, np.array([9, 15]))
np.testing.assert_allclose(column_means, np.array([2.5, 3.5, 6.0]))
np.testing.assert_array_equal(row_argmax, np.array([1, 2]))
```

### P09：布尔索引与输入保护

文件：`p09_boolean_masks.py`

实现：

```python
import numpy as np

def values_between(values, low, high):
    ...

def replace_outside_range(values, low, high):
    ...
```

要求：

- `values_between` 返回位于闭区间 `[low, high]` 内的元素。
- `replace_outside_range` 返回新数组：小于 `low` 的值改为 `low`，大于 `high` 的值改为 `high`。
- `replace_outside_range` 使用布尔索引，不直接调用 `np.clip`。
- 两个函数都不能修改输入数组。

自测：

```python
values = np.array([-2, 0, 3, 8])
original = values.copy()
np.testing.assert_array_equal(values_between(values, 0, 3), np.array([0, 3]))
np.testing.assert_array_equal(
    replace_outside_range(values, 0, 5),
    np.array([0, 0, 3, 5]),
)
np.testing.assert_array_equal(values, original)
```

## 第三阶段：广播与向量化

### P10：按列标准化

文件：`p10_standardize.py`

实现：

```python
import numpy as np

def standardize_columns(matrix):
    ...
```

要求：

- 对每一列使用 `(x - mean) / std`。
- 使用 `axis=0` 和广播，不使用循环。
- 如果某列标准差为 `0`，该列结果全部设为 `0`。
- 返回浮点数组，不修改输入。

自测：

```python
matrix = np.array([[1, 10, 5], [3, 20, 5], [5, 30, 5]])
original = matrix.copy()
result = standardize_columns(matrix)
np.testing.assert_allclose(result.mean(axis=0), np.array([0.0, 0.0, 0.0]), atol=1e-7)
np.testing.assert_allclose(result[:, 2], np.zeros(3))
np.testing.assert_array_equal(matrix, original)
```

### P11：按行 min-max 归一化

文件：`p11_row_normalize.py`

实现：

```python
import numpy as np

def normalize_rows(matrix):
    ...
```

要求：

- 每行独立使用 `(x - row_min) / (row_max - row_min)`。
- 使用 `keepdims=True` 保持可广播的 shape。
- 如果一行的最大值等于最小值，该行全部返回 `0`。
- 不使用循环。

自测：

```python
matrix = np.array([[1, 2, 3], [5, 5, 5], [10, 0, 5]])
expected = np.array([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.5]])
np.testing.assert_allclose(normalize_rows(matrix), expected)
```

### P12：添加 bias 列

文件：`p12_add_bias.py`

实现：

```python
import numpy as np

def add_bias_column(features):
    ...
```

要求：

- 输入 shape 为 `(N, D)`。
- 在最后添加一列全为 `1` 的数据，返回 shape `(N, D + 1)`。
- 使用 `np.ones` 和数组拼接，不使用循环。

自测：

```python
features = np.array([[2, 3], [4, 5]])
expected = np.array([[2, 3, 1], [4, 5, 1]])
np.testing.assert_array_equal(add_bias_column(features), expected)
assert add_bias_column(features).shape == (2, 3)
```

### P13：NumPy one-hot

文件：`p13_one_hot_numpy.py`

实现：

```python
import numpy as np

def one_hot(labels, num_classes):
    ...
```

要求：

- 返回 shape 为 `(样本数, num_classes)` 的整数数组。
- 先创建零数组，再使用 NumPy 高级索引一次设置所有 `1`。
- 不使用 Python 循环，也暂时不使用 `np.eye`。
- 标签越界或 `num_classes <= 0` 时返回 `None`。

自测：

```python
labels = np.array([2, 0, 1])
expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
np.testing.assert_array_equal(one_hot(labels, 3), expected)
assert one_hot(np.array([], dtype=int), 3).shape == (0, 3)
assert one_hot(np.array([3]), 3) is None
```

### P14：类别预测与准确率

文件：`p14_prediction_metrics.py`

实现：

```python
import numpy as np

def predict_classes(logits):
    ...

def accuracy(predictions, labels):
    ...
```

要求：

- `logits` shape 为 `(样本数, 类别数)`。
- `predict_classes` 返回每行最大值的索引。
- `accuracy` 使用向量比较和平均值。
- 长度不同返回 `None`，两个空数组返回 `0.0`。
- 不使用循环。

自测：

```python
logits = np.array([[0.1, 0.7, 0.2], [5.0, 1.0, 2.0]])
predictions = predict_classes(logits)
np.testing.assert_array_equal(predictions, np.array([1, 0]))
assert accuracy(predictions, np.array([1, 2])) == 0.5
assert accuracy(np.array([], dtype=int), np.array([], dtype=int)) == 0.0
assert accuracy(np.array([1]), np.array([1, 0])) is None
```

### P15：成对平方距离

文件：`p15_pairwise_distances.py`

实现：

```python
import numpy as np

def pairwise_squared_distances(points_a, points_b):
    ...
```

要求：

- `points_a` shape 为 `(N, D)`，`points_b` shape 为 `(M, D)`。
- 返回 shape 为 `(N, M)`，其中每项是两个点之间的平方距离。
- 使用增加维度、广播、逐元素平方和 `axis` 完成。
- 不使用循环。
- 两组点的特征维度不同则返回 `None`。

自测：

```python
a = np.array([[0, 0], [1, 1]])
b = np.array([[1, 0], [3, 4]])
expected = np.array([[1, 25], [1, 13]])
np.testing.assert_array_equal(pairwise_squared_distances(a, b), expected)
assert pairwise_squared_distances(np.zeros((2, 3)), np.zeros((2, 4))) is None
```

### P16：NumPy 版 1-NN

文件：`p16_nearest_neighbor_numpy.py`

实现：

```python
import numpy as np

def predict_1nn(train_features, train_labels, test_features):
    ...
```

要求：

- 可以导入并调用 P15 的函数。
- 一次计算所有测试样本到训练样本的距离。
- 使用 `argmin(axis=1)` 找最近训练样本。
- 不使用循环。
- 训练特征和标签数量不同，或训练集为空时返回 `None`。

自测：

```python
train_x = np.array([[0, 0], [3, 3]])
train_y = np.array([0, 1])
test_x = np.array([[1, 0], [2.5, 2.5]])
np.testing.assert_array_equal(predict_1nn(train_x, train_y, test_x), np.array([0, 1]))
assert predict_1nn(np.empty((0, 2)), np.array([]), test_x) is None
```

## 第四阶段：矩阵乘法与分类函数

### P17：线性层前向传播

文件：`p17_linear_forward.py`

实现：

```python
import numpy as np

def linear_forward(features, weights, bias):
    ...
```

要求：

- `features` shape 为 `(N, D)`。
- `weights` shape 为 `(D, C)`。
- `bias` shape 为 `(C,)`。
- 返回 `features @ weights + bias`，shape 为 `(N, C)`。
- 理解 bias 是如何广播到所有样本的。

自测：

```python
features = np.array([[1.0, 2.0], [3.0, 4.0]])
weights = np.array([[1.0, 0.0], [0.0, 2.0]])
bias = np.array([0.5, -1.0])
expected = np.array([[1.5, 3.0], [3.5, 7.0]])
np.testing.assert_allclose(linear_forward(features, weights, bias), expected)
```

### P18：数值稳定的 softmax

文件：`p18_softmax.py`

实现：

```python
import numpy as np

def softmax(logits):
    ...
```

要求：

- 对二维 `logits` 的每一行计算 softmax。
- 指数运算前，每行减去该行最大值，避免溢出。
- 分母求和时使用 `keepdims=True`。
- 不使用循环。

自测：

```python
logits = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
probabilities = softmax(logits)
np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
np.testing.assert_allclose(probabilities[0], probabilities[1])
assert np.all(np.isfinite(probabilities))
```

### P19：交叉熵损失

文件：`p19_cross_entropy.py`

实现：

```python
import numpy as np

def cross_entropy(probabilities, labels):
    ...
```

要求：

- `probabilities` shape 为 `(N, C)`，`labels` shape 为 `(N,)`。
- 用高级索引取出每个样本真实类别的概率。
- 返回负对数概率的平均值。
- 取对数前用一个很小的正数保护概率下限。
- 样本数量不匹配返回 `None`；空输入返回 `0.0`。

自测：

```python
probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
labels = np.array([0, 2])
np.testing.assert_allclose(cross_entropy(probabilities, labels), -np.log(0.7))
assert cross_entropy(np.empty((0, 3)), np.array([], dtype=int)) == 0.0
assert cross_entropy(np.ones((2, 2)) / 2, np.array([0])) is None
```

### P20：可复现的随机 mini-batch

文件：`p20_shuffled_batches.py`

实现：

```python
import numpy as np

def make_shuffled_batches(features, labels, batch_size, seed):
    ...
```

要求：

- 使用 `np.random.default_rng(seed)` 创建随机数生成器。
- 使用同一组随机索引打乱特征和标签。
- 返回由 `(x_batch, y_batch)` 组成的列表。
- 相同输入和相同 seed 必须产生相同顺序。
- 长度不同或 `batch_size <= 0` 时返回 `None`。

自测：

```python
features = np.array([[10], [20], [30], [40], [50]])
labels = np.array([1, 2, 3, 4, 5])
batches_a = make_shuffled_batches(features, labels, 2, seed=42)
batches_b = make_shuffled_batches(features, labels, 2, seed=42)

x_a = np.concatenate([batch[0] for batch in batches_a])
y_a = np.concatenate([batch[1] for batch in batches_a])
x_b = np.concatenate([batch[0] for batch in batches_b])
y_b = np.concatenate([batch[1] for batch in batches_b])

np.testing.assert_array_equal(x_a, x_b)
np.testing.assert_array_equal(y_a, y_b)
np.testing.assert_array_equal(x_a[:, 0] / 10, y_a)
assert [len(batch[0]) for batch in batches_a] == [2, 2, 1]
```

## 第五阶段：梯度与线性分类器训练

### P21：softmax 交叉熵梯度

文件：`p21_softmax_gradient.py`

实现：

```python
import numpy as np

def softmax_cross_entropy_with_gradient(logits, labels):
    ...
```

要求：

- 返回 `(loss, gradient)`。
- `loss` 是平均交叉熵。
- 先计算 softmax 概率，再复制概率数组。
- 每个样本真实类别位置减去 `1`，最后除以样本数，得到对 `logits` 的梯度。
- 不能修改 softmax 概率数组。

自测：

```python
logits = np.zeros((2, 3))
labels = np.array([0, 2])
loss, gradient = softmax_cross_entropy_with_gradient(logits, labels)

np.testing.assert_allclose(loss, np.log(3.0))
expected_gradient = np.array([
    [-1 / 3, 1 / 6, 1 / 6],
    [1 / 6, 1 / 6, -1 / 3],
])
np.testing.assert_allclose(gradient, expected_gradient)
np.testing.assert_allclose(gradient.sum(axis=1), np.zeros(2))
```

### P22：线性分类器的参数梯度

文件：`p22_linear_gradients.py`

实现：

```python
import numpy as np

def linear_classifier_gradients(features, weights, bias, labels):
    ...
```

要求：

- 可以调用 P17 和 P21 的函数。
- 返回 `(loss, weight_gradient, bias_gradient)`。
- `weight_gradient = features.T @ logits_gradient`。
- `bias_gradient` 对所有样本的 logits 梯度按行求和。
- 返回的梯度 shape 必须分别与 `weights`、`bias` 相同。

自测：

```python
features = np.eye(2)
weights = np.zeros((2, 2))
bias = np.zeros(2)
labels = np.array([0, 1])

loss, weight_gradient, bias_gradient = linear_classifier_gradients(
    features, weights, bias, labels
)
np.testing.assert_allclose(loss, np.log(2.0))
np.testing.assert_allclose(
    weight_gradient,
    np.array([[-0.25, 0.25], [0.25, -0.25]]),
)
np.testing.assert_allclose(bias_gradient, np.zeros(2))
assert weight_gradient.shape == weights.shape
assert bias_gradient.shape == bias.shape
```

### P23：一次梯度下降更新

文件：`p23_train_step.py`

实现：

```python
import numpy as np

def train_step(features, labels, weights, bias, learning_rate):
    ...
```

要求：

- 调用 P22 获得损失与梯度。
- 使用 `parameter - learning_rate * gradient` 更新参数。
- 返回 `(new_weights, new_bias, loss_before_update)`。
- 不允许原地修改传入的 `weights` 和 `bias`。

自测：

```python
from p17_linear_forward import linear_forward
from p18_softmax import softmax
from p19_cross_entropy import cross_entropy

features = np.eye(2)
labels = np.array([0, 1])
weights = np.zeros((2, 2))
bias = np.zeros(2)
weights_before = weights.copy()
bias_before = bias.copy()

new_weights, new_bias, old_loss = train_step(
    features, labels, weights, bias, learning_rate=1.0
)
new_loss = cross_entropy(softmax(linear_forward(features, new_weights, new_bias)), labels)

assert new_loss < old_loss
np.testing.assert_array_equal(weights, weights_before)
np.testing.assert_array_equal(bias, bias_before)
```

### P24：训练一个 softmax 线性分类器

文件：`p24_train_classifier.py`

实现：

```python
import numpy as np

def train_linear_classifier(
    features,
    labels,
    num_classes,
    epochs,
    learning_rate,
):
    ...
```

要求：

- 权重初始化为 shape `(特征数, num_classes)` 的零数组。
- bias 初始化为 shape `(num_classes,)` 的零数组。
- 每轮使用全部训练数据执行一次 P23 的更新。
- 每轮更新后重新计算 loss 和 accuracy。
- 返回 `(weights, bias, history)`。
- `history` 每项格式为 `{"epoch": 1, "loss": ..., "accuracy": ...}`。
- epoch 从 `1` 开始；`history` 长度等于 `epochs`。

自测：

```python
features = np.array([
    [2.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 2.0],
])
labels = np.array([0, 0, 1, 1])

weights, bias, history = train_linear_classifier(
    features,
    labels,
    num_classes=2,
    epochs=20,
    learning_rate=0.5,
)

assert weights.shape == (2, 2)
assert bias.shape == (2,)
assert len(history) == 20
assert history[0]["epoch"] == 1
assert history[-1]["epoch"] == 20
assert history[-1]["loss"] < history[0]["loss"]
assert history[-1]["accuracy"] == 1.0
```

## 完成本阶段后的能力目标

完成 P01-P24 后，你应该能够：

- 用 shape 判断机器学习数据的组织方式。
- 熟练使用索引、切片、布尔掩码、`axis` 和广播。
- 把纯 Python 循环改写成 NumPy 向量化计算。
- 实现 one-hot、归一化、距离矩阵、mini-batch 和分类准确率。
- 理解线性层、softmax、交叉熵和参数梯度之间的关系。
- 用 NumPy 训练一个最小可用的多分类线性模型。

下一阶段将进入：MNIST 数据读取、两层神经网络、ReLU、反向传播、训练与模型评估。
