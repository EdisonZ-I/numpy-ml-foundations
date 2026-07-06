# NumPy and ML Foundations Practice

目标：用尽量少而必要的综合练习掌握 NumPy、向量化和神经网络训练核心，为下一步直接实现 MNIST 分类器做准备。

练习原则：

- 每次完成 1 到 3 题。
- 每题会同时训练多个相关知识点，不重复安排只为熟练度服务的相似题。
- 先独立实现，再运行基础断言；批改时还会加入边界测试。
- P02 之后，除非题目允许，避免用 Python 循环处理样本或特征维度。
- 不修改输入数组，除非题目明确要求。

环境准备：

```powershell
python -m pip install -r requirements.txt
python -c "import numpy as np; print(np.__version__)"
```

测试约定：

- 整数数组：`np.testing.assert_array_equal`
- 浮点数组：`np.testing.assert_allclose`
- 所有文件中的 NumPy 统一写作 `import numpy as np`

## 第一阶段：数据管线与 NumPy 思维

### P01：纯 Python 配对数据管线

文件：`p01_dataset_pipeline_python.py`

实现：

```python
def prepare_dataset(features, labels, test_ratio, batch_size):
    ...
```

要求：

- 本题只使用纯 Python。
- `features` 和 `labels` 必须保持一一对应。
- 测试集数量为 `int(len(features) * test_ratio)`，取数据末尾。
- 训练集按 `batch_size` 切分，返回由 `(x_batch, y_batch)` 组成的列表。
- 返回字典：`{"train_batches": ..., "test": (x_test, y_test)}`。
- 长度不同、`test_ratio` 不在 `[0, 1]` 内或 `batch_size <= 0` 时返回 `None`。
- 正确处理测试集数量为 `0` 的情况。

自测：

```python
result = prepare_dataset(
    [[1], [2], [3], [4], [5]],
    [10, 20, 30, 40, 50],
    test_ratio=0.4,
    batch_size=2,
)
assert result == {
    "train_batches": [([[1], [2]], [10, 20]), ([[3]], [30])],
    "test": ([[4], [5]], [40, 50]),
}

result = prepare_dataset([[1], [2]], [10, 20], 0, 3)
assert result == {"train_batches": [([[1], [2]], [10, 20])], "test": ([], [])}
assert prepare_dataset([[1]], [], 0.2, 2) is None
```

### P02：图像数组的 shape、reshape 与 axis

文件：`p02_image_batch_summary.py`

实现：

```python
import numpy as np

def image_batch_summary(images):
    ...
```

要求：

- 用 `np.asarray` 接收列表或数组。
- 输入必须为 `(N, H, W)` 三维数据，否则返回 `None`。
- 将图像展平为 `(N, H * W)`。
- 计算每张图像所有像素之和。
- 返回每张图像最大像素在展平后的位置；并列时取第一个。
- 返回包含 `original_shape`、`flat`、`pixel_sums`、`brightest_indices` 的字典。
- 不使用循环。

自测：

```python
images = np.array([
    [[1, 2], [3, 4]],
    [[9, 0], [2, 1]],
])
summary = image_batch_summary(images)
assert summary["original_shape"] == (2, 2, 2)
np.testing.assert_array_equal(summary["flat"], np.array([[1, 2, 3, 4], [9, 0, 2, 1]]))
np.testing.assert_array_equal(summary["pixel_sums"], np.array([10, 12]))
np.testing.assert_array_equal(summary["brightest_indices"], np.array([3, 0]))
assert image_batch_summary(np.array([1, 2, 3])) is None
```

### P03：无数据泄漏的像素预处理

文件：`p03_pixel_preprocessing.py`

实现：

```python
import numpy as np

def preprocess_pixels(train_features, test_features):
    ...
```

要求：

- 输入均为 `(N, D)`，特征数不同则返回 `None`。
- 转成浮点数并除以 `255.0`。
- 只使用训练集计算每列的 `mean` 和 `std`，测试集必须复用这些统计量。
- 标准差为 `0` 的列使用 `1.0` 作为除数，使该列结果为 `0`。
- 返回 `(processed_train, processed_test, mean, scale)`。
- 不使用循环，也不修改输入。

自测：

```python
train = np.array([[0, 255], [255, 255]])
test = np.array([[127.5, 255]])
train_before = train.copy()

processed_train, processed_test, mean, scale = preprocess_pixels(train, test)
np.testing.assert_allclose(processed_train, np.array([[-1.0, 0.0], [1.0, 0.0]]))
np.testing.assert_allclose(processed_test, np.array([[0.0, 0.0]]))
np.testing.assert_allclose(mean, np.array([0.5, 1.0]))
np.testing.assert_allclose(scale, np.array([0.5, 1.0]))
np.testing.assert_array_equal(train, train_before)
```

## 第二阶段：广播、标签与批次

### P04：向量化距离矩阵与 1-NN

文件：`p04_vectorized_knn.py`

实现：

```python
import numpy as np

def pairwise_squared_distances(test_features, train_features):
    ...

def predict_1nn(train_features, train_labels, test_features):
    ...
```

要求：

- 距离函数返回 shape `(测试样本数, 训练样本数)`。
- 使用增加维度、广播、逐元素平方和 `axis`，不能使用循环。
- 特征维度不同返回 `None`。
- `predict_1nn` 用 `argmin(axis=1)` 找最近样本；并列时取训练集中第一个。
- 训练特征与标签数量不同或训练集为空时返回 `None`。

自测：

```python
train_x = np.array([[0, 0], [3, 3]])
train_y = np.array([0, 1])
test_x = np.array([[1, 0], [2, 2]])

np.testing.assert_array_equal(
    pairwise_squared_distances(test_x, train_x),
    np.array([[1, 13], [8, 2]]),
)
np.testing.assert_array_equal(predict_1nn(train_x, train_y, test_x), np.array([0, 1]))
```

### P05：标签编码与完整分类评估

文件：`p05_classification_metrics.py`

实现：

```python
import numpy as np

def one_hot(labels, num_classes):
    ...

def evaluate_logits(logits, labels):
    ...
```

要求：

- `one_hot` 创建零数组后，用高级索引一次写入所有 `1`；不使用循环或 `np.eye`。
- 标签越界或 `num_classes <= 0` 时返回 `None`。
- `evaluate_logits` 返回 `predictions`、`accuracy`、`confusion_matrix`。
- 混淆矩阵的行是真实类别，列是预测类别。
- 使用 `np.add.at` 累加混淆矩阵，不使用循环。
- logits 与 labels 样本数不同返回 `None`。

自测：

```python
labels = np.array([2, 0, 1])
np.testing.assert_array_equal(
    one_hot(labels, 3),
    np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
)

logits = np.array([[0.1, 0.9], [3.0, 1.0], [2.0, 4.0]])
result = evaluate_logits(logits, np.array([1, 1, 0]))
np.testing.assert_array_equal(result["predictions"], np.array([1, 0, 1]))
assert result["accuracy"] == 1 / 3
np.testing.assert_array_equal(result["confusion_matrix"], np.array([[0, 1], [1, 1]]))
```

### P06：可复现的随机 mini-batch

文件：`p06_shuffled_batches.py`

实现：

```python
import numpy as np

def make_shuffled_batches(features, labels, batch_size, seed):
    ...
```

要求：

- 使用 `np.random.default_rng(seed).permutation(...)`。
- 同一组索引必须同时用于特征和标签。
- 返回由 `(x_batch, y_batch)` 组成的列表。
- 相同输入和 seed 必须得到相同顺序。
- 长度不同或 `batch_size <= 0` 时返回 `None`。

自测：

```python
features = np.array([[10], [20], [30], [40], [50]])
labels = np.array([1, 2, 3, 4, 5])
batches_a = make_shuffled_batches(features, labels, 2, seed=42)
batches_b = make_shuffled_batches(features, labels, 2, seed=42)

x_a = np.concatenate([x for x, _ in batches_a])
y_a = np.concatenate([y for _, y in batches_a])
x_b = np.concatenate([x for x, _ in batches_b])
y_b = np.concatenate([y for _, y in batches_b])

np.testing.assert_array_equal(x_a, x_b)
np.testing.assert_array_equal(y_a, y_b)
np.testing.assert_array_equal(x_a[:, 0] / 10, y_a)
assert [len(x) for x, _ in batches_a] == [2, 2, 1]
```

## 第三阶段：线性分类器与梯度

### P07：稳定的线性分类前向管线

文件：`p07_linear_classifier.py`

实现：

```python
import numpy as np

def softmax(logits):
    ...

def linear_classifier_loss(features, labels, weights, bias):
    ...
```

要求：

- logits 使用 `features @ weights + bias`。
- softmax 对每行计算，指数运算前每行减去最大值。
- 用高级索引取真实类别概率并计算平均交叉熵。
- 对数前将概率下限保护为 `1e-12`。
- `linear_classifier_loss` 返回 `(loss, probabilities, predictions)`。
- 不使用循环。

自测：

```python
features = np.eye(2)
labels = np.array([0, 1])
weights = np.zeros((2, 2))
bias = np.zeros(2)

loss, probabilities, predictions = linear_classifier_loss(features, labels, weights, bias)
np.testing.assert_allclose(loss, np.log(2.0))
np.testing.assert_allclose(probabilities, np.full((2, 2), 0.5))
np.testing.assert_array_equal(predictions, np.array([0, 0]))

stable = softmax(np.array([[1000.0, 1001.0, 1002.0]]))
assert np.all(np.isfinite(stable))
np.testing.assert_allclose(stable.sum(axis=1), np.ones(1))
```

### P08：通用数值梯度检查器

文件：`p08_numerical_gradient.py`

实现：

```python
import numpy as np

def numerical_gradient(loss_function, parameter, epsilon=1e-5):
    ...
```

要求：

- `loss_function(parameter)` 返回一个标量损失。
- 对参数中的每个元素使用中心差分：`(loss_plus - loss_minus) / (2 * epsilon)`。
- 本题允许使用循环或 `np.ndindex`。
- 计算结束后，`parameter` 必须恢复原值。
- 返回与参数 shape 相同的梯度数组。

自测：

```python
parameter = np.array([[1.0, -2.0], [3.0, 0.5]])
before = parameter.copy()
gradient = numerical_gradient(lambda value: np.sum(value ** 2), parameter)
np.testing.assert_allclose(gradient, 2 * parameter, rtol=1e-5, atol=1e-5)
np.testing.assert_array_equal(parameter, before)
```

### P09：线性分类器解析梯度

文件：`p09_linear_gradients.py`

实现：

```python
import numpy as np

def linear_loss_and_gradients(features, labels, weights, bias):
    ...
```

要求：

- 返回 `(loss, weight_gradient, bias_gradient)`。
- softmax 概率复制后，在真实类别位置减 `1`，再除以样本数，得到 logits 梯度。
- `weight_gradient = features.T @ logits_gradient`。
- `bias_gradient` 是 logits 梯度沿样本轴求和。
- 不使用循环。
- 用 P08 的数值梯度验证解析梯度，而不是只检查 shape。

自测：

```python
from p08_numerical_gradient import numerical_gradient

features = np.array([[1.0, 2.0], [-1.0, 1.0]])
labels = np.array([0, 1])
weights = np.array([[0.1, -0.2], [0.3, 0.2]])
bias = np.array([0.05, -0.05])

loss, weight_gradient, bias_gradient = linear_loss_and_gradients(
    features, labels, weights, bias
)
numerical_w = numerical_gradient(
    lambda value: linear_loss_and_gradients(features, labels, value, bias)[0],
    weights,
)
numerical_b = numerical_gradient(
    lambda value: linear_loss_and_gradients(features, labels, weights, value)[0],
    bias,
)

np.testing.assert_allclose(weight_gradient, numerical_w, rtol=1e-4, atol=1e-5)
np.testing.assert_allclose(bias_gradient, numerical_b, rtol=1e-4, atol=1e-5)
```

### P10：训练 softmax 线性分类器

文件：`p10_train_linear_classifier.py`

实现：

```python
import numpy as np

def train_linear_classifier(
    features,
    labels,
    num_classes,
    epochs,
    learning_rate,
    batch_size,
    seed,
):
    ...
```

要求：

- 权重和 bias 初始化为零。
- 每轮用 `seed + epoch` 调用 P06 获得随机 batch，并调用 P09 计算每个 batch 的梯度。
- 按 `parameter -= learning_rate * gradient` 更新参数。
- 每轮结束后在完整训练集上计算 loss 和 accuracy。
- 返回 `(weights, bias, history)`。
- `history` 每项为 `{"epoch": ..., "loss": ..., "accuracy": ...}`。

自测：

```python
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

assert weights.shape == (2, 2)
assert bias.shape == (2,)
assert len(history) == 40
assert history[-1]["loss"] < history[0]["loss"]
assert history[-1]["accuracy"] == 1.0
```

## 第四阶段：两层神经网络

### P11：ReLU 两层网络前向传播

文件：`p11_two_layer_forward.py`

实现：

```python
import numpy as np

def relu(values):
    ...

def two_layer_forward(features, weight1, bias1, weight2, bias2):
    ...
```

要求：

- 第一层：`hidden_linear = features @ weight1 + bias1`。
- 激活：`hidden = relu(hidden_linear)`。
- 第二层：`logits = hidden @ weight2 + bias2`。
- 返回 `(logits, cache)`。
- `cache` 至少保存反向传播需要的 `features`、`hidden_linear` 和 `hidden`。
- 不使用循环。

自测：

```python
features = np.array([[2.0, 1.0]])
weight1 = np.array([[1.0, -1.0], [0.5, 1.0]])
bias1 = np.zeros(2)
weight2 = np.array([[1.0, -1.0], [2.0, 0.5]])
bias2 = np.array([0.1, 0.2])

logits, cache = two_layer_forward(features, weight1, bias1, weight2, bias2)
np.testing.assert_allclose(logits, np.array([[2.6, -2.3]]))
np.testing.assert_allclose(cache["hidden_linear"], np.array([[2.5, -1.0]]))
np.testing.assert_allclose(cache["hidden"], np.array([[2.5, 0.0]]))
```

### P12：两层网络反向传播与梯度检查

文件：`p12_two_layer_gradients.py`

实现：

```python
import numpy as np

def two_layer_loss_and_gradients(
    features,
    labels,
    weight1,
    bias1,
    weight2,
    bias2,
):
    ...
```

要求：

- 返回 `(loss, gradients)`，其中 `gradients` 包含四个参数的梯度。
- 先完成 P11 前向传播和 softmax 交叉熵。
- 从 logits 梯度开始反向计算：第二层参数梯度、隐藏层梯度、ReLU 梯度、第一层参数梯度。
- ReLU 在 `hidden_linear <= 0` 的位置梯度为 `0`。
- 不使用循环计算解析梯度。
- 至少使用 P08 对 `weight1` 和 `weight2` 做数值梯度检查。

自测：

```python
from p08_numerical_gradient import numerical_gradient

features = np.array([[1.0, 2.0], [-1.0, -2.0]])
labels = np.array([0, 1])
weight1 = np.array([[0.2, -0.3], [0.4, 0.1]])
bias1 = np.array([0.1, 0.2])
weight2 = np.array([[0.5, -0.2], [-0.1, 0.3]])
bias2 = np.zeros(2)

loss, gradients = two_layer_loss_and_gradients(
    features, labels, weight1, bias1, weight2, bias2
)
numerical_w1 = numerical_gradient(
    lambda value: two_layer_loss_and_gradients(
        features, labels, value, bias1, weight2, bias2
    )[0],
    weight1,
)
numerical_w2 = numerical_gradient(
    lambda value: two_layer_loss_and_gradients(
        features, labels, weight1, bias1, value, bias2
    )[0],
    weight2,
)

np.testing.assert_allclose(gradients["weight1"], numerical_w1, rtol=1e-4, atol=1e-5)
np.testing.assert_allclose(gradients["weight2"], numerical_w2, rtol=1e-4, atol=1e-5)
assert gradients["bias1"].shape == bias1.shape
assert gradients["bias2"].shape == bias2.shape
```

### P13：训练非线性分类器

文件：`p13_train_two_layer_network.py`

实现：

```python
import numpy as np

def train_two_layer_network(
    features,
    labels,
    hidden_size,
    num_classes,
    epochs,
    learning_rate,
    seed,
):
    ...
```

要求：

- 使用 `np.random.default_rng(seed)` 初始化权重。
- 两层权重从均值 `0`、标准差 `0.5` 的正态分布采样，bias 初始化为零。
- 每轮在完整数据上调用 P12，更新四个参数。
- 每轮更新后记录 loss 和 accuracy。
- 返回 `(parameters, history)`；`parameters` 是包含四个参数的字典。
- 使用下面的非线性数据验证两层网络确实学到了线性模型无法表达的分类边界。

自测：

```python
features = np.array([
    [-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0],
    [-2.0, 2.0], [-1.0, 1.0], [1.0, -1.0], [2.0, -2.0],
])
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

parameters, history = train_two_layer_network(
    features,
    labels,
    hidden_size=8,
    num_classes=2,
    epochs=2000,
    learning_rate=0.1,
    seed=0,
)

assert parameters["weight1"].shape == (2, 8)
assert parameters["weight2"].shape == (8, 2)
assert len(history) == 2000
assert history[-1]["loss"] < history[0]["loss"]
assert history[-1]["accuracy"] == 1.0
```

## 完成本阶段后的能力目标

完成这 13 题后，你应当能够：

- 理解并操作 `(N, D)`、`(N, H, W)`、`(D, C)` 等核心 shape。
- 使用 reshape、axis、广播、高级索引和向量化替代样本级循环。
- 完成数据预处理、随机 batch、分类评估和距离分类。
- 实现稳定 softmax、交叉熵、数值梯度和解析梯度检查。
- 从零训练 softmax 线性分类器和 ReLU 两层神经网络。

下一阶段直接进入 MNIST：读取数据、训练两层网络、验证集评估、保存参数，然后用 PyTorch 重写。
