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

- 每个练习文件在自测部分从 `test_helpers.py` 导入需要的检查函数。
- `check_equal` 检查普通 Python 值，`check_array_equal` 检查整数数组，`check_allclose` 检查浮点数组。
- 测试成功会打印 `[PASS] 测试名称`。
- 测试失败会打印测试名称、期望值和实际值，然后保留 `AssertionError` 和 traceback。
- 所有文件中的 NumPy 统一写作 `import numpy as np`

格式记号：

- `list[T]`：元素类型为 `T` 的 Python 列表。
- `dict[str, T]`：键为字符串、值类型为 `T` 的 Python 字典。
- `np.ndarray, shape=(N, D)`：有 `N` 行样本、每个样本有 `D` 个特征的二维数组。
- `N` 表示样本数，`D` 表示特征数，`C` 表示类别数，`H/W` 表示图像高度和宽度。

## 第一阶段：数据管线与 NumPy 思维

### P01：纯 Python 配对数据管线

文件：`p01_dataset_pipeline_python.py`

**使用场景**

监督学习数据通常由样本 `features` 和标签 `labels` 配对组成。训练前需要从末尾留出测试集，再把训练部分切成小批次；整个过程中不能让某个样本和它的标签错位。

**输入格式**

- `features: list[list[number]]`：长度为 `N`，每个元素是一条样本。
- `labels: list[int]`：长度为 `N`，`labels[i]` 属于 `features[i]`。
- `test_ratio: float`：测试集比例，范围 `[0, 1]`。
- `batch_size: int`：每个训练 batch 的最大样本数。

**转换规则**

先根据 `test_ratio` 把 `features`、`labels` 同步切成训练部分和测试部分，再根据 `batch_size` 把训练部分同步分批，最终组合成一个结果字典。

**样例输入**

```python
features = [[1], [2], [3], [4], [5]]
labels = [10, 20, 30, 40, 50]
test_ratio = 0.4
batch_size = 2
```

**样例输出**

```python
{
    "train_batches": [
        ([[1], [2]], [10, 20]),
        ([[3]], [30]),
    ],
    "test": ([[4], [5]], [40, 50]),
}
```

**需要实现**

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
from test_helpers import check_equal, check_is_none

check_equal(
    "P01/正常切分并分批",
    prepare_dataset(
        [[1], [2], [3], [4], [5]],
        [10, 20, 30, 40, 50],
        test_ratio=0.4,
        batch_size=2,
    ),
    {
        "train_batches": [([[1], [2]], [10, 20]), ([[3]], [30])],
        "test": ([[4], [5]], [40, 50]),
    },
)
check_equal(
    "P01/测试集比例为零",
    prepare_dataset([[1], [2]], [10, 20], 0, 3),
    {"train_batches": [([[1], [2]], [10, 20])], "test": ([], [])},
)
check_is_none(
    "P01/特征与标签长度不同",
    prepare_dataset([[1]], [], 0.2, 2),
)
```

### P02：图像数组的 shape、reshape 与 axis

文件：`p02_image_batch_summary.py`

**使用场景**

一批灰度图像常以三维数组 `images` 保存。模型通常需要二维的扁平特征，同时我们也可能需要快速获得每张图像的像素总量和最亮像素位置，以检查数据是否合理。

**输入格式**

- `images: np.ndarray, shape=(N, H, W)`：`N` 张高为 `H`、宽为 `W` 的灰度图像。

**转换规则**

把 `images` 从 `(N, H, W)` 变为 `flat: (N, H*W)`，再沿每个样本的特征轴计算 `pixel_sums` 和 `brightest_indices`，最后与原 shape 一起放进字典。

**样例输入**

```python
from test_helpers import check_array_equal, check_equal, check_is_none

images = np.array([
    [[1, 2], [3, 4]],
    [[9, 0], [2, 1]],
])
```

**样例输出**

```python
{
    "original_shape": (2, 2, 2),
    "flat": np.array([[1, 2, 3, 4], [9, 0, 2, 1]]),
    "pixel_sums": np.array([10, 12]),
    "brightest_indices": np.array([3, 0]),
}
```

**需要实现**

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
check_equal("P02/保留原始shape", summary["original_shape"], (2, 2, 2))
check_array_equal(
    "P02/展平图像",
    summary["flat"],
    np.array([[1, 2, 3, 4], [9, 0, 2, 1]]),
)
check_array_equal("P02/逐图像像素和", summary["pixel_sums"], np.array([10, 12]))
check_array_equal("P02/最亮像素索引", summary["brightest_indices"], np.array([3, 0]))
check_is_none("P02/拒绝非三维输入", image_batch_summary(np.array([1, 2, 3])))
```

### P03：无数据泄漏的像素预处理

文件：`p03_pixel_preprocessing.py`

**使用场景**

图像像素通常位于 `0` 到 `255`。训练前需要缩放和标准化，但测试集不能参与统计量计算，否则会发生数据泄漏。因此只用 `train_features` 计算均值和标准差，再用相同参数处理 `test_features`。

**输入格式**

- `train_features: np.ndarray, shape=(N_train, D)`：训练图像的扁平特征。
- `test_features: np.ndarray, shape=(N_test, D)`：测试图像的扁平特征。

**转换规则**

先把两个数组转成浮点数并除以 `255.0`。只从缩放后的 `train_features` 得到每列的 `mean` 和 `scale`，然后分别计算 `(features - mean) / scale`。零标准差应替换为 `1.0`。

**样例输入**

```python
train_features = np.array([[0, 255], [255, 255]])
test_features = np.array([[127.5, 255]])
```

**样例输出**

```python
(
    np.array([[-1.0, 0.0], [1.0, 0.0]]),  # processed_train
    np.array([[0.0, 0.0]]),               # processed_test
    np.array([0.5, 1.0]),                  # mean
    np.array([0.5, 1.0]),                  # scale
)
```

**需要实现**

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
```

## 第二阶段：广播、标签与批次

### P04：向量化距离矩阵与 1-NN

文件：`p04_vectorized_knn.py`

**使用场景**

1-NN 分类需要比较每个测试样本和所有训练样本。逐对写循环会很慢，因此要用广播一次生成距离矩阵，再从每行找到最近训练样本的标签。

**输入格式**

- `train_features: np.ndarray, shape=(N_train, D)`：已知类别的训练样本。
- `train_labels: np.ndarray, shape=(N_train,)`：训练样本标签。
- `test_features: np.ndarray, shape=(N_test, D)`：需要预测的测试样本。

**转换规则**

将 `test_features` 和 `train_features` 广播为所有点对的差，沿特征轴求平方和，得到 `distances: (N_test, N_train)`。每行最小值的列索引对应最近训练样本，再用该索引读取 `train_labels`。

**样例输入**

```python
train_features = np.array([[0, 0], [3, 3]])
train_labels = np.array([0, 1])
test_features = np.array([[1, 0], [2, 2]])
```

**样例输出**

```python
distances = np.array([[1, 13], [8, 2]])
predictions = np.array([0, 1])
```

**需要实现**

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
```

### P05：标签编码与完整分类评估

文件：`p05_classification_metrics.py`

**使用场景**

训练分类模型时，整数标签 `labels` 有时需要转换为 one-hot；模型输出 `logits` 后，还需要统一得到预测类别、准确率和混淆矩阵，才能判断具体哪些类别被混淆。

**输入格式**

- `labels: np.ndarray, shape=(N,)`：每项是范围 `[0, C-1]` 内的整数类别。
- `num_classes: int`：类别总数 `C`。
- `logits: np.ndarray, shape=(N, C)`：模型对每个样本、每个类别给出的分数。

**转换规则**

`one_hot` 将 `labels[i]` 变成长度为 `num_classes`、仅对应位置为 `1` 的向量。`evaluate_logits` 对 `logits` 每行取最大值索引得到预测，再与 `labels` 比较并累计混淆矩阵。

**样例输入**

```python
labels = np.array([1, 1, 0])
logits = np.array([[0.1, 0.9], [3.0, 1.0], [2.0, 4.0]])
```

**样例输出**

```python
one_hot(labels, 2) == np.array([[0, 1], [0, 1], [1, 0]])

evaluate_logits(logits, labels) == {
    "predictions": np.array([1, 0, 1]),
    "accuracy": 1 / 3,
    "confusion_matrix": np.array([[0, 1], [1, 1]]),
}
```

说明：NumPy 数组不能直接用整个字典的 `==` 判断，上面只是输出结构示意，实际测试见后面的断言。

**需要实现**

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
```

### P06：可复现的随机 mini-batch

文件：`p06_shuffled_batches.py`

**使用场景**

训练时通常要在每轮打乱样本并制作 mini-batch，同时必须保持样本与标签对应。为了调试和复现实验，同一个 `seed` 还必须产生相同顺序。

**输入格式**

- `features: np.ndarray, shape=(N, D)`：全部样本特征。
- `labels: np.ndarray, shape=(N,)`：全部样本标签。
- `batch_size: int`：每批最大样本数。
- `seed: int`：控制随机排列的种子。

**转换规则**

用 `seed` 生成包含 `0` 到 `N-1` 的随机索引，把同一索引同时应用到 `features` 和 `labels`，再沿样本轴按 `batch_size` 切分。

**样例输入**

```python
from test_helpers import check_array_equal, check_equal

features = np.array([[10], [20], [30], [40], [50]])
labels = np.array([1, 2, 3, 4, 5])
batch_size = 2
seed = 42
```

**样例输出**

```python
[
    (np.array([[50], [30]]), np.array([5, 3])),
    (np.array([[40], [20]]), np.array([4, 2])),
    (np.array([[10]]), np.array([1])),
]
```

**需要实现**

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

check_array_equal("P06/相同seed产生相同特征顺序", x_a, x_b)
check_array_equal("P06/相同seed产生相同标签顺序", y_a, y_b)
check_array_equal("P06/打乱后特征标签仍配对", x_a[:, 0] / 10, y_a)
check_equal("P06/batch大小包含末尾余数", [len(x) for x, _ in batches_a], [2, 2, 1])
```

## 第三阶段：线性分类器与梯度

### P07：稳定的线性分类前向管线

文件：`p07_linear_classifier.py`

**使用场景**

线性分类器先把 `features` 映射成类别分数 `logits`，再用 softmax 转成概率。训练时需要根据真实 `labels` 计算交叉熵损失，预测时则取概率最大的类别。

**输入格式**

- `features: np.ndarray, shape=(N, D)`：一批样本。
- `labels: np.ndarray, shape=(N,)`：真实类别。
- `weights: np.ndarray, shape=(D, C)`：从特征到类别的权重。
- `bias: np.ndarray, shape=(C,)`：每个类别的偏置。
- `logits: np.ndarray, shape=(N, C)`：softmax 单独调用时的类别分数。

**转换规则**

先计算 `logits = features @ weights + bias`，对每行 logits 做稳定 softmax 得到 `probabilities`，再取真实类别概率计算平均负对数损失，并以每行最大概率索引作为 `predictions`。

**样例输入**

```python
from test_helpers import check_allclose, check_array_equal, check_true

features = np.eye(2)
labels = np.array([0, 1])
weights = np.zeros((2, 2))
bias = np.zeros(2)
```

**样例输出**

```python
(
    np.log(2.0),                  # loss
    np.full((2, 2), 0.5),         # probabilities
    np.array([0, 0]),             # predictions
)
```

**需要实现**

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
```

### P08：通用数值梯度检查器

文件：`p08_numerical_gradient.py`

**使用场景**

手写反向传播后，很难只靠观察判断梯度是否正确。数值梯度会轻微增加和减少 `parameter` 中的每个元素，观察 `loss_function` 的变化，用来检查解析梯度。

**输入格式**

- `loss_function: callable`：接收 `parameter` 并返回标量损失的函数。
- `parameter: np.ndarray, 任意 shape`：需要检查梯度的参数。
- `epsilon: float`：每次扰动参数的微小数值，默认 `1e-5`。

**转换规则**

对 `parameter` 的每个位置分别计算 `loss_plus` 和 `loss_minus`，使用中心差分得到该位置梯度。每次计算后必须恢复原参数，最终返回同 shape 的梯度数组。

**样例输入**

```python
from test_helpers import check_allclose, check_array_equal

parameter = np.array([[1.0, -2.0], [3.0, 0.5]])
loss_function = lambda value: np.sum(value ** 2)
```

**样例输出**

```python
np.array([[2.0, -4.0], [6.0, 1.0]])
```

**需要实现**

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
check_allclose(
    "P08/二次函数数值梯度",
    gradient,
    2 * parameter,
    rtol=1e-5,
    atol=1e-5,
)
check_array_equal("P08/计算后恢复原参数", parameter, before)
```

### P09：线性分类器解析梯度

文件：`p09_linear_gradients.py`

**使用场景**

P08 可以验证梯度，但逐元素扰动太慢，不能真正用于训练。线性分类器需要根据矩阵求导规则，一次算出 `weights` 和 `bias` 的解析梯度，再用数值梯度确认它们正确。

**输入格式**

- `features: np.ndarray, shape=(N, D)`：训练样本。
- `labels: np.ndarray, shape=(N,)`：真实类别。
- `weights: np.ndarray, shape=(D, C)`：线性层权重。
- `bias: np.ndarray, shape=(C,)`：线性层偏置。

**转换规则**

先完成线性前向传播与 softmax 交叉熵，再从概率得到 `logits_gradient: (N, C)`。利用矩阵乘法把它传播到 `weight_gradient: (D, C)`，沿样本轴求和得到 `bias_gradient: (C,)`。

**样例输入**

```python
features = np.array([[1.0, 2.0], [-1.0, 1.0]])
labels = np.array([0, 1])
weights = np.array([[0.1, -0.2], [0.3, 0.2]])
bias = np.array([0.05, -0.05])
```

**样例输出**

```python
(
    0.5409423053,
    np.array([[-0.41468225, 0.41468225],
              [-0.11683329, 0.11683329]]),
    np.array([0.06033856, -0.06033856]),
)
```

浮点结果允许有微小误差，以后面的 `assert_allclose` 为准。

**需要实现**

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
from test_helpers import check_allclose

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

check_allclose(
    "P09/weight解析梯度匹配数值梯度",
    weight_gradient,
    numerical_w,
    rtol=1e-4,
    atol=1e-5,
)
check_allclose(
    "P09/bias解析梯度匹配数值梯度",
    bias_gradient,
    numerical_b,
    rtol=1e-4,
    atol=1e-5,
)
```

### P10：训练 softmax 线性分类器

文件：`p10_train_linear_classifier.py`

**使用场景**

有了数据 batch 和解析梯度，就可以执行完整训练：重复打乱数据、计算每批梯度、更新参数，并在每轮结束后记录模型是否真的在进步。

**输入格式**

- `features: np.ndarray, shape=(N, D)`：训练特征。
- `labels: np.ndarray, shape=(N,)`：训练标签。
- `num_classes: int`：类别数 `C`。
- `epochs: int`：完整遍历训练集的轮数。
- `learning_rate: float`：每次更新的步长。
- `batch_size: int`：每个 batch 的最大样本数。
- `seed: int`：随机打乱的基础种子。

**转换规则**

初始化 `weights: (D, C)` 和 `bias: (C,)`。每轮用 `seed + epoch` 制作 batch，对每批计算梯度并更新参数；随后用完整 `features` 计算本轮 loss 和 accuracy，写入 `history`。

**样例输入**

```python
from test_helpers import check_equal, check_true

features = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
labels = np.array([0, 0, 1, 1])
num_classes = 2
epochs = 40
learning_rate = 0.5
batch_size = 2
seed = 0
```

**样例输出**

```python
weights ≈ np.array([[ 1.6094, -1.6094],
                    [-1.5965,  1.5965]])
bias ≈ np.array([-0.0075, 0.0075])
history[0]  ≈ {"epoch": 1,  "loss": 0.3276, "accuracy": 1.0}
history[-1] ≈ {"epoch": 40, "loss": 0.0207, "accuracy": 1.0}
```

浮点值用于帮助理解输出，不要求逐位相同；结构、趋势和后面的断言才是判断标准。

**需要实现**

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
```

## 第四阶段：两层神经网络

### P11：ReLU 两层网络前向传播

文件：`p11_two_layer_forward.py`

**使用场景**

线性分类器只能学习一条线性边界。为了表达更复杂关系，需要先把 `features` 送入第一层并经过 ReLU，再由第二层产生类别 logits。反向传播还需要保存部分中间结果。

**输入格式**

- `features: np.ndarray, shape=(N, D)`：输入样本。
- `weight1: np.ndarray, shape=(D, H)`、`bias1: np.ndarray, shape=(H,)`：第一层参数。
- `weight2: np.ndarray, shape=(H, C)`、`bias2: np.ndarray, shape=(C,)`：第二层参数。
- `H` 表示隐藏单元数。

**转换规则**

计算 `hidden_linear = features @ weight1 + bias1`，把其中负数通过 ReLU 变成 `0` 得到 `hidden`，再计算 `logits = hidden @ weight2 + bias2`。返回 logits 以及保存中间变量的 `cache`。

**样例输入**

```python
from test_helpers import check_allclose

features = np.array([[2.0, 1.0]])
weight1 = np.array([[1.0, -1.0], [0.5, 1.0]])
bias1 = np.zeros(2)
weight2 = np.array([[1.0, -1.0], [2.0, 0.5]])
bias2 = np.array([0.1, 0.2])
```

**样例输出**

```python
logits = np.array([[2.6, -2.3]])
cache = {
    "features": features,
    "hidden_linear": np.array([[2.5, -1.0]]),
    "hidden": np.array([[2.5, 0.0]]),
}
```

`cache` 可以额外保存其他必要内容，但至少要包含以上三项。

**需要实现**

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
check_allclose("P11/两层网络logits", logits, np.array([[2.6, -2.3]]))
check_allclose(
    "P11/第一层线性输出",
    cache["hidden_linear"],
    np.array([[2.5, -1.0]]),
)
check_allclose("P11/ReLU隐藏层输出", cache["hidden"], np.array([[2.5, 0.0]]))
```

### P12：两层网络反向传播与梯度检查

文件：`p12_two_layer_gradients.py`

**使用场景**

两层网络训练时，损失首先影响第二层 logits，然后影响隐藏层，最后影响第一层参数。这个过程就是反向传播；由于步骤更多，需要用数值梯度检查实现是否正确。

**输入格式**

- `features: np.ndarray, shape=(N, D)`、`labels: np.ndarray, shape=(N,)`：训练数据。
- `weight1: np.ndarray, shape=(D, H)`、`bias1: np.ndarray, shape=(H,)`：第一层参数。
- `weight2: np.ndarray, shape=(H, C)`、`bias2: np.ndarray, shape=(C,)`：第二层参数。

**转换规则**

先得到 loss 和 logits 梯度。梯度按 `logits -> hidden -> hidden_linear -> features` 的反方向传播：先求第二层参数梯度，再乘 `weight2.T`，经过 ReLU 的梯度掩码，最后求第一层参数梯度。

**样例输入**

```python
features = np.array([[1.0, 2.0], [-1.0, -2.0]])
labels = np.array([0, 1])
weight1 = np.array([[0.2, -0.3], [0.4, 0.1]])
bias1 = np.array([0.1, 0.2])
weight2 = np.array([[0.5, -0.2], [-0.1, 0.3]])
bias2 = np.zeros(2)
```

**样例输出**

```python
loss ≈ 0.5141386082
gradients = {
    "weight1": np.array([[-0.11381815, 0.15904614],
                         [-0.22763631, 0.31809227]]),
    "bias1": np.array([-0.11381815, -0.02896824]),
    "weight2": np.array([[-0.17885710, 0.17885710],
                         [ 0.05424566, -0.05424566]]),
    "bias2": np.array([0.07242061, -0.07242061]),
}
```

浮点结果允许有微小误差，以数值梯度检查为准。

**需要实现**

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
from test_helpers import check_allclose, check_equal

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

check_allclose(
    "P12/weight1反向传播匹配数值梯度",
    gradients["weight1"],
    numerical_w1,
    rtol=1e-4,
    atol=1e-5,
)
check_allclose(
    "P12/weight2反向传播匹配数值梯度",
    gradients["weight2"],
    numerical_w2,
    rtol=1e-4,
    atol=1e-5,
)
check_equal("P12/bias1梯度shape", gradients["bias1"].shape, bias1.shape)
check_equal("P12/bias2梯度shape", gradients["bias2"].shape, bias2.shape)
```

### P13：训练非线性分类器

文件：`p13_train_two_layer_network.py`

**使用场景**

最后把初始化、前向传播、反向传播、参数更新和训练日志组合起来，训练一个能处理非线性边界的小网络。这是进入 MNIST 前对整条 NumPy 神经网络训练链路的总检查。

**输入格式**

- `features: np.ndarray, shape=(N, D)`、`labels: np.ndarray, shape=(N,)`：训练数据。
- `hidden_size: int`：隐藏单元数 `H`。
- `num_classes: int`：类别数 `C`。
- `epochs: int`、`learning_rate: float`：训练轮数和更新步长。
- `seed: int`：参数初始化的随机种子。

**转换规则**

根据 `D/H/C` 初始化四个参数。每轮用 P12 计算 loss 和全部梯度，更新参数后重新计算预测与准确率并写入 `history`。最终返回参数字典和完整训练记录。

**样例输入**

```python
from test_helpers import check_equal, check_true

features = np.array([
    [-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0],
    [-2.0,  2.0], [-1.0,  1.0], [1.0, -1.0], [2.0, -2.0],
])
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
hidden_size = 8
num_classes = 2
epochs = 2000
learning_rate = 0.1
seed = 0
```

**样例输出**

```python
parameters = {
    "weight1": <np.ndarray shape=(2, 8)>,
    "bias1": <np.ndarray shape=(8,)>,
    "weight2": <np.ndarray shape=(8, 2)>,
    "bias2": <np.ndarray shape=(2,)>,
}
history = [
    {"epoch": 1, "loss": <float>, "accuracy": <float>},
    ...,
    {"epoch": 2000, "loss": 约0.004, "accuracy": 1.0},
]
```

尖括号内容是结构说明，不是需要写进 Python 的代码。

**需要实现**

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
```

## 完成本阶段后的能力目标

完成这 13 题后，你应当能够：

- 理解并操作 `(N, D)`、`(N, H, W)`、`(D, C)` 等核心 shape。
- 使用 reshape、axis、广播、高级索引和向量化替代样本级循环。
- 完成数据预处理、随机 batch、分类评估和距离分类。
- 实现稳定 softmax、交叉熵、数值梯度和解析梯度检查。
- 从零训练 softmax 线性分类器和 ReLU 两层神经网络。

下一阶段直接进入 MNIST：读取数据、训练两层网络、验证集评估、保存参数，然后用 PyTorch 重写。
