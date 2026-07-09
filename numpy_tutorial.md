# NumPy 入门：从纯 Python 数组思维到机器学习向量化

这份教程默认你已经掌握：

- Python 基本语法、函数、列表、字典、切片和异常处理；
- 基础算法与文件读取；
- 用嵌套列表手写过向量和矩阵操作；
- 理解标签（label）、one-hot、准确率（accuracy）和 batch；
- 但还没有系统使用 NumPy。

这里的目标不是背函数，而是建立一个稳定的 NumPy 心智模型：

> 数据是什么形状？每个轴代表什么？这一步沿哪个轴计算？输出形状是什么？

这四个问题会贯穿之后的线性回归、逻辑回归、softmax、KNN、MNIST、MLP 和 PyTorch。

---

## 学习方法与路线

不要一口气把全文当作 API 手册读完。每次只完成一节：

1. 先读“直觉”；
2. 手算示例的 shape 和结果；
3. 再运行最小代码；
4. 完成 TODO；
5. 用断言验证后再进入下一节。

推荐顺序：

```text
环境与导入
  -> ndarray、shape、dtype
  -> 索引与切片
  -> 向量化与逐元素运算
  -> reshape 与轴
  -> 聚合运算
  -> broadcasting
  -> 矩阵乘法
  -> 副本与视图
  -> 随机数和 batch
  -> 机器学习常见写法
```

---

## 第 0 节：准备环境

### 1. 当前目标

能确认当前 Python 解释器，并成功导入 NumPy。

### 2. 需要理解的概念

NumPy 是第三方库，不属于 Python 标准库。安装包时，包会被安装到某个具体的 Python 环境中。因此：

```text
安装 NumPy 的 Python
必须和
运行代码的 Python
是同一个解释器
```

`python -m pip` 的含义是：让当前这个 `python` 去运行它自己的 `pip` 模块。它比直接写 `pip` 更不容易装错环境。

### 3. 最小操作骨架

在 `numpy-ml-foundations` 目录打开 PowerShell：

```powershell
python -m pip install -r requirements.txt
python -c "import sys; print(sys.executable)"
python -c "import numpy as np; print(np.__version__)"
```

代码中统一使用约定写法：

```python
import numpy as np
```

`np` 只是模块别名，不是特殊语法。

### 4. 如何验证

第二条命令应输出正在使用的 Python 路径，第三条命令应输出 NumPy 版本且没有异常。

### 5. 常见错误

- `ModuleNotFoundError: No module named 'numpy'`：当前解释器没有安装 NumPy。
- 编辑器能导入、终端不能导入：编辑器和终端使用了不同解释器。
- 虚拟环境无法启动：虚拟环境记录的原 Python 路径可能已经失效，需要重新创建环境。
- 不要把自己的文件命名为 `numpy.py`，否则可能遮蔽真正的 NumPy 包。

### 6. 完成后发什么检查

发这两条命令的输出：解释器路径和 NumPy 版本。

---

## 第 1 节：`ndarray` 是什么

### 1. 当前目标

能创建数组，并读懂 `shape`、`ndim`、`size` 和 `dtype`。

### 2. 需要理解的概念

#### 直觉：NumPy 数组不是“更短的列表”

Python 列表可以混合保存不同类型的对象：

```python
values = [1, "hello", [2, 3]]
```

NumPy 的核心对象是 `ndarray`，即 N-dimensional array。它通常把大量**同一种数据类型**的值按规则形状组织起来。

```python
import numpy as np

x = np.array([[1, 2, 3],
              [4, 5, 6]])
```

可以把它想成：

```text
数据缓冲区 + shape + dtype
```

这里：

```text
shape = (2, 3)  两行三列
ndim  = 2       有两个轴
size  = 6       一共有六个元素
dtype = 整数类型
```

#### 形状的数学含义

若矩阵记作：

$$X \in \mathbb{R}^{N \times D}$$

符号含义：

| 数学符号 | 含义 | 代码对应 |
|---|---|---|
| $X$ | 整个数据矩阵 | `x` |
| $N$ | 样本数量 | `x.shape[0]` |
| $D$ | 每个样本的特征数量 | `x.shape[1]` |
| $\mathbb{R}$ | 元素是实数 | 通常使用浮点 `dtype` |

机器学习里通常约定：

```text
行 = 样本
列 = 特征
```

例如 100 张已经展平的 MNIST 图片：

```text
X.shape == (100, 784)
```

#### 四个基本属性

```python
print(x.shape)  # (2, 3)
print(x.ndim)   # 2
print(x.size)   # 6
print(x.dtype)  # 具体名称与系统有关
```

注意：`shape` 是元组。`(3,)` 表示一维数组，不是三行一列的二维矩阵。

### 3. 最小代码骨架

```python
import numpy as np

x = np.array([
    [10, 20, 30],
    [40, 50, 60],
])

# TODO：分别打印 x 的类型、shape、ndim、size、dtype
```

常见创建方式：

```python
np.array([1, 2, 3])          # 根据已有数据创建
np.asarray([1, 2, 3])        # 尽量转换成数组，已有数组时通常避免不必要复制
np.zeros((2, 3))             # 全 0
np.ones((2, 3))              # 全 1
np.full((2, 3), 7)           # 全 7
np.arange(0, 10, 2)          # 类似 range，得到 0, 2, 4, 6, 8
np.linspace(0.0, 1.0, 5)     # 在 0 到 1 之间均匀取 5 个点
```

`np.zeros((2, 3))` 有两层括号：外层是函数调用，内层 `(2, 3)` 是 shape 元组。

### 4. 如何验证

```python
assert x.shape == (2, 3)
assert x.ndim == 2
assert x.size == 6
```

### 5. 常见错误

- 把 `shape` 和 `size` 混淆：`shape` 描述各轴长度，`size` 是总元素数。
- 认为 `(3,)` 和 `(3, 1)` 相同：前者一维，后者二维列向量。
- 写 `np.zeros(2, 3)`：第二个位置参数不是第二个维度；shape 应写成元组。
- 创建不规则嵌套列表，例如 `[[1, 2], [3]]`：它不是规则矩阵。

### 6. 完成后发什么检查

发你的代码和五项输出，并用一句话解释 `(2, 3)` 中的 `2` 与 `3`。

---

## 第 2 节：索引、切片与布尔筛选

### 1. 当前目标

能从二维数组中选出单个元素、一行、一列、子矩阵，以及满足条件的元素。

### 2. 需要理解的概念

二维数组的索引格式：

```python
x[行索引, 列索引]
```

数组如下：

```python
x = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90],
])
```

基本索引：

```python
x[1, 2]      # 第 1 行、第 2 列 -> 60
x[1]         # 第 1 行，shape 是 (3,)
x[:, 1]      # 所有行的第 1 列，shape 是 (3,)
x[0:2, 1:3]  # 前两行、后两列，shape 是 (2, 2)
```

`:` 表示这个轴全部保留。

#### 保留维度

```python
x[:, 1].shape    # (3,)
x[:, 1:2].shape  # (3, 1)
```

整数索引 `1` 会取掉对应的轴；切片 `1:2` 会保留这个轴。这一点会影响后面的 broadcasting 和矩阵乘法。

#### 布尔筛选

```python
mask = x >= 50
selected = x[mask]
```

`mask` 是与 `x` 同形状的布尔数组：

```text
False False False
False True  True
True  True  True
```

`x[mask]` 会取出所有为 `True` 的位置，并返回一维结果。

这不是 Python 的单个 `bool`，而是对每个元素分别比较。

#### 高级索引

```python
rows = np.array([2, 0])
picked = x[rows]
```

含义是按顺序取第 2 行和第 0 行。之后 one-hot 和打乱数据都要使用这种“数组作为索引”的能力。

### 3. 最小代码骨架

```python
import numpy as np

x = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90],
])

# TODO 1：取第二行
# TODO 2：取第三列，并保留为 shape (3, 1)
# TODO 3：取右下角 2x2 子矩阵
# TODO 4：取出所有能被 20 整除的元素
```

多个条件不能使用 Python 的 `and`、`or`：

```python
mask = (x >= 30) & (x <= 70)
```

每个条件都要加括号，因为比较运算和 `&` 的优先级容易产生误解。

### 4. 如何验证

```python
np.testing.assert_array_equal(
    x[0:2, 1:3],
    np.array([[20, 30], [50, 60]]),
)
assert x[:, 2:3].shape == (3, 1)
```

### 5. 常见错误

- 写 `x[1][2]` 虽然通常能运行，但 `x[1, 2]` 更直接地表达多轴索引。
- 用 `and` 连接数组条件，会出现“数组真值不明确”的异常。
- 忘记切片右边界不包含在结果中。
- 认为布尔筛选会保留原 shape；它通常返回一维数组。

### 6. 完成后发什么检查

发四个 TODO 的表达式、结果和 shape。

---

## 第 3 节：向量化与逐元素运算

### 1. 当前目标

把“逐个元素循环”改写为数组整体运算，并区分逐元素乘法与矩阵乘法。

### 2. 需要理解的概念

#### 直觉

纯 Python 写法常常是：

```python
result = []
for value in values:
    result.append(value * 2 + 1)
```

NumPy 写法是：

```python
result = values * 2 + 1
```

这叫向量化（vectorization）：描述“对整个数组做什么”，由 NumPy 在底层执行批量计算。

它的优势不只是代码短：

- 运算含义更接近数学公式；
- 大量循环在高效的底层代码中完成；
- 更容易扩展到一个 batch 的样本。

#### 逐元素公式

若：

$$y_i = 2x_i + 1$$

符号对应：

| 数学符号 | 含义 | 代码对应 |
|---|---|---|
| $x_i$ | `x` 的第 `i` 个元素 | `x[i]` |
| $y_i$ | 输出的第 `i` 个元素 | `y[i]` |
| $2x_i+1$ | 每个元素采用同一规则 | `x * 2 + 1` |

```python
x = np.array([1, 2, 3])
y = x * 2 + 1
```

常见逐元素运算：

```python
x + y
x - y
x * y
x / y
x ** 2
np.sqrt(x)
np.exp(x)
np.log(x)
np.maximum(x, 0)
```

#### `*` 与 `@`

```python
a * b  # 对应位置逐元素相乘
a @ b  # 线性代数中的矩阵乘法
```

这两个操作绝不能混为一谈。矩阵乘法会在第 7 节单独讲。

### 3. 最小代码骨架

```python
import numpy as np

pixels = np.array([0, 64, 128, 255])

# TODO：不用循环，把像素缩放到 [0, 1]
normalized = ...

# TODO：对每个元素计算 x^2 + 2x + 1
x = np.array([-2.0, 0.0, 3.0])
y = ...
```

### 4. 如何验证

```python
np.testing.assert_allclose(
    normalized,
    np.array([0.0, 64 / 255, 128 / 255, 1.0]),
)
np.testing.assert_allclose(y, (x + 1) ** 2)
```

浮点数验证推荐 `np.testing.assert_allclose`，不要依赖严格的 `==`。

### 5. 常见错误

- 对 Python 列表写 `values * 2`：列表会重复两遍，不会逐元素乘 2。
- 用 `*` 代替矩阵乘法。
- 为了“使用 NumPy”仍然逐个索引写 Python 循环，失去了向量化意义。
- 浮点计算用 `assert a == b`，可能受舍入误差影响。

### 6. 完成后发什么检查

发两个 TODO 的表达式和断言运行结果。

---

## 第 4 节：`reshape`、转置与增加维度

### 1. 当前目标

能改变数组形状，并在操作前后追踪每个轴的含义。

### 2. 需要理解的概念

#### `reshape` 只改变看法，不改变元素数量

```python
x = np.arange(12)
y = x.reshape(3, 4)
```

```text
x.shape == (12,)
y.shape == (3, 4)
```

元素总数必须一致：

$$12 = 3 \times 4$$

可以用 `-1` 让 NumPy 自动推断一个维度：

```python
x.reshape(3, -1)  # 自动得到 (3, 4)
```

一个 shape 中最多只能有一个 `-1`。

#### 图像展平

若一个 batch 的灰度图形状是：

```text
images.shape == (N, H, W)
```

符号含义：

| 符号 | 含义 | 代码位置 |
|---|---|---|
| $N$ | 图片数量 | `images.shape[0]` |
| $H$ | 每张图高度 | `images.shape[1]` |
| $W$ | 每张图宽度 | `images.shape[2]` |

展平每张图片时，希望输出：

$$X \in \mathbb{R}^{N \times (H W)}$$

代码骨架：

```python
flat = images.reshape(images.shape[0], -1)
```

关键是保留第 0 轴的样本含义，只合并后面的图像轴。

#### 转置

```python
x.T
```

二维数组从 `(N, D)` 变成 `(D, N)`。对于一维数组：

```python
v = np.array([1, 2, 3])
v.T.shape  # 仍然是 (3,)
```

一维数组没有“行轴和列轴”可以交换。若需要列向量：

```python
column = v.reshape(-1, 1)
# 或
column = v[:, None]
```

#### `None` / `np.newaxis`

它们用于在指定位置增加长度为 1 的轴：

```python
v[None, :].shape  # (1, 3)，行向量
v[:, None].shape  # (3, 1)，列向量
```

### 3. 最小代码骨架

```python
import numpy as np

images = np.arange(24).reshape(2, 3, 4)

# TODO 1：展平每张图片，得到 (2, 12)
flat = ...

features = np.array([10, 20, 30])

# TODO 2：变成 shape (1, 3)
row = ...

# TODO 3：变成 shape (3, 1)
column = ...
```

### 4. 如何验证

```python
assert flat.shape == (2, 12)
assert row.shape == (1, 3)
assert column.shape == (3, 1)
np.testing.assert_array_equal(flat[0], np.arange(12))
```

### 5. 常见错误

- 展平整个 batch 为 `(N * H * W,)`，丢失样本边界。
- 认为 `v.T` 能把一维数组变成列向量。
- `reshape` 后元素数不一致。
- 只看元素，不检查 shape；很多 bug 的数值看似合理，形状却已经错了。

### 6. 完成后发什么检查

发三个 TODO 和三个输出 shape。

---

## 第 5 节：`axis` 与聚合运算

### 1. 当前目标

真正理解 `axis`，能正确计算每列、每行或每个样本的统计量。

### 2. 需要理解的概念

`sum`、`mean`、`max`、`min`、`argmax`、`std` 等函数都经常接收 `axis`。

最稳的理解不是“axis=0 是列运算”，而是：

> `axis=k` 表示沿第 `k` 个轴压缩；这个轴会被汇总掉。

设：

```python
x = np.array([
    [1, 2, 3],
    [4, 5, 6],
])
# shape: (2, 3)
```

#### 不指定 `axis`

```python
x.sum()  # 21
```

所有元素汇总为一个值。

#### `axis=0`

```python
x.sum(axis=0)  # [5, 7, 9]，shape (3,)
```

第 0 轴长度是 2。把两行沿第 0 轴压缩后，第 0 轴消失，只剩 3 列。

#### `axis=1`

```python
x.sum(axis=1)  # [6, 15]，shape (2,)
```

第 1 轴长度是 3。每一行的三个元素被压缩，第 1 轴消失。

#### 公式与代码对应

若 $X \in \mathbb{R}^{N \times D}$，每个样本的特征和为：

$$s_i = \sum_{j=1}^{D} X_{ij}$$

| 符号 | 含义 | 代码对应 |
|---|---|---|
| $i$ | 样本索引 | 第 0 轴 |
| $j$ | 特征索引 | 第 1 轴 |
| $X_{ij}$ | 第 `i` 个样本的第 `j` 个特征 | `x[i, j]` |
| 对 $j$ 求和 | 汇总特征轴 | `x.sum(axis=1)` |
| $s$ | 每个样本一个结果 | shape `(N,)` |

#### `argmax`

`max` 返回最大值，`argmax` 返回最大值所在的索引：

```python
logits = np.array([
    [0.1, 2.0, 0.3],
    [3.0, 1.0, 0.5],
])

predictions = logits.argmax(axis=1)  # [1, 0]
```

这里每行是一个样本，每列是一个类别，所以要压缩类别轴 `axis=1`。

#### `keepdims=True`

```python
x.mean(axis=0).shape                 # (3,)
x.mean(axis=0, keepdims=True).shape  # (1, 3)
```

`keepdims=True` 不移除被压缩的轴，而是将其长度保留为 1。这样常常更容易继续 broadcasting。

### 3. 最小代码骨架

```python
import numpy as np

scores = np.array([
    [0.2, 0.7, 0.1],
    [0.8, 0.1, 0.1],
    [0.3, 0.2, 0.5],
])

# TODO 1：每个样本的总分
# TODO 2：每个类别在所有样本中的平均分
# TODO 3：每个样本分数最高的类别索引
```

写代码前先预测三个结果的 shape。

### 4. 如何验证

```python
np.testing.assert_allclose(scores.sum(axis=1), np.ones(3))
np.testing.assert_array_equal(scores.argmax(axis=1), np.array([1, 0, 2]))
```

### 5. 常见错误

- 死记“axis=0 按列”，到三维数组就失效。
- logits 形状为 `(N, C)` 时使用 `argmax(axis=0)`，结果会变成每个类别一个样本索引。
- 忘记聚合通常会让一个轴消失。
- `argmax()` 不传 axis，得到的是整个数组展平后的一个索引。

### 6. 完成后发什么检查

发三个表达式，并在每个表达式前写出你预测的输出 shape。

---

## 第 6 节：Broadcasting 广播

### 1. 当前目标

能判断两个不同 shape 的数组是否可以一起逐元素运算，并预测结果 shape。

### 2. 需要理解的概念

#### 直觉

假设每个样本有三个特征：

```python
x.shape    == (4, 3)
mean.shape == (3,)
```

希望每个样本都减去同一组特征均值：

```python
centered = x - mean
```

NumPy 不需要你把 `mean` 手动复制成四行。它会逻辑上让 `(3,)` 对齐到 `(4, 3)` 的最后一轴，这就是 broadcasting。

#### 广播规则

从 shape 的**最右边**开始逐轴比较。两个维度兼容，当且仅当：

1. 两者相等；或
2. 其中一个等于 1；或
3. 某个数组缺少这一轴，可把缺少部分看作 1。

例子：

```text
(4, 3)
   (3,)
-------
(4, 3)  可以
```

```text
(4, 3)
(4, 1)
-------
(4, 3)  可以
```

```text
(4, 3)
   (4,)
-------
不可以，因为最右边 3 和 4 不相等，且都不是 1
```

#### 标准化公式

机器学习中常见的特征标准化：

$$Z_{ij} = \frac{X_{ij} - \mu_j}{\sigma_j}$$

| 符号 | 含义 | shape | 代码对应 |
|---|---|---|---|
| $X$ | 输入数据 | `(N, D)` | `x` |
| $X_{ij}$ | 样本 `i` 的特征 `j` | 标量 | `x[i, j]` |
| $\mu_j$ | 第 `j` 个特征的均值 | `(D,)` | `mean` |
| $\sigma_j$ | 第 `j` 个特征的标准差 | `(D,)` | `std` |
| $Z$ | 标准化结果 | `(N, D)` | `z` |

```python
mean = x.mean(axis=0)
std = x.std(axis=0)
z = (x - mean) / std
```

`mean` 和 `std` 都是 `(D,)`，会广播到每个样本。

#### 两两距离中的广播

训练样本与测试样本：

```text
test.shape  = (T, D)
train.shape = (R, D)
```

增加维度后：

```text
test[:, None, :].shape = (T, 1, D)
train[None, :, :].shape = (1, R, D)
```

相减结果：

```text
(T, R, D)
```

它表示：每个测试样本、每个训练样本、每个特征上的差。这是向量化 KNN 的核心形状。

### 3. 最小代码骨架

```python
import numpy as np

x = np.array([
    [1.0, 10.0, 100.0],
    [3.0, 14.0, 200.0],
])

# TODO 1：计算每列均值
mean = ...

# TODO 2：利用广播让每一列减去自己的均值
centered = ...

# TODO 3：先写出以下表达式的结果 shape，再运行确认
a = np.zeros((5, 1, 3))
b = np.zeros((1, 4, 3))
c = a + b
```

### 4. 如何验证

```python
assert mean.shape == (3,)
assert centered.shape == x.shape
np.testing.assert_allclose(centered.mean(axis=0), np.zeros(3))
assert c.shape == (5, 4, 3)
```

### 5. 常见错误

- 从左边开始比较 shape；广播必须从最右边开始。
- 把 `(N,)` 误当成 `(N, 1)`。
- 代码能广播就认为含义正确；错误的 shape 有时也能合法广播，却在错误的轴上重复。
- 标准化时标准差为 0，导致除零；常见做法是把对应除数替换为 `1.0`。

### 6. 完成后发什么检查

发 `mean`、`centered`、`c` 的 shape，并解释 `(5, 1, 3) + (1, 4, 3)` 为什么得到 `(5, 4, 3)`。

---

## 第 7 节：矩阵乘法与线性模型

### 1. 当前目标

理解 `@` 的形状规则，并把线性模型公式准确翻译为 NumPy。

### 2. 需要理解的概念

#### 单个样本

线性模型：

$$z = \sum_{j=1}^{D} x_j w_j + b$$

也可写为：

$$z = xw + b$$

若：

```text
x.shape == (D,)
w.shape == (D,)
```

则：

```python
z = x @ w + b
```

`x @ w` 是向量内积，输出标量。

#### 一个 batch、一个输出

$$z_i = \sum_{j=1}^{D} X_{ij}w_j + b$$

shape 推导：

```text
X: (N, D)
w: (D,)
----------- @
z: (N,)
```

代码：

```python
z = x @ w + b
```

每个样本得到一个输出。

#### 一个 batch、多个类别

线性分类器：

$$Z = XW + b$$

shape：

```text
X: (N, D)  N 个样本，每个 D 个特征
W: (D, C)  每个特征到 C 个类别的权重
b: (C,)    每个类别一个偏置
-------------------------------
Z: (N, C)  每个样本对每个类别的分数
```

代码：

```python
logits = x @ w + b
```

这里发生两件事：

1. `x @ w` 做矩阵乘法，得到 `(N, C)`；
2. `b` 通过 broadcasting 加到每一行，结果仍为 `(N, C)`。

#### 元素公式与代码对应

$$Z_{ic} = \sum_{j=1}^{D} X_{ij}W_{jc} + b_c$$

| 符号 | 含义 | 代码对应 |
|---|---|---|
| $i$ | 第几个样本 | `x`、`logits` 的第 0 轴 |
| $j$ | 第几个输入特征 | `x` 的第 1 轴、`w` 的第 0 轴 |
| $c$ | 第几个类别 | `w`、`b`、`logits` 的类别轴 |
| $\sum_j$ | 输入特征维度上的乘积求和 | `x @ w` |

矩阵乘法的核心 shape 规则：

```text
(N, D) @ (D, C) -> (N, C)
```

中间两个 `D` 必须相等，并在乘法中被消去。

### 3. 最小代码骨架

```python
import numpy as np

x = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
])                    # (N=3, D=2)

w = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
])                    # (D=2, C=2)

b = np.array([1.0, -1.0])  # (C=2,)

# TODO：计算每个样本的两个类别分数
logits = ...
```

### 4. 如何验证

```python
assert logits.shape == (3, 2)

# 用第一行手算，验证矩阵乘法到底做了什么
expected_first = np.array([
    1.0 * 0.1 + 2.0 * 0.3 + 1.0,
    1.0 * 0.2 + 2.0 * 0.4 - 1.0,
])
np.testing.assert_allclose(logits[0], expected_first)
```

### 5. 常见错误

- 用 `x * w` 代替 `x @ w`。
- 把权重写成 `(C, D)`，却仍直接计算 `x @ w`。
- 不写 shape 推导，等报错后反复试转置。
- 一维与二维数组混用后，误以为 `.T` 一定会改变方向。

### 6. 完成后发什么检查

发 `logits` 的代码、shape、第一行结果，以及你的 shape 推导。

---

## 第 8 节：`dtype`、类型转换与数值问题

### 1. 当前目标

知道数组的数据类型为什么重要，能在需要时安全转换为浮点数。

### 2. 需要理解的概念

NumPy 数组通常使用统一的 `dtype`：

```python
x = np.array([1, 2, 3], dtype=np.int64)
y = np.array([1, 2, 3], dtype=np.float64)
```

常见类型：

```text
int32 / int64      整数
float32 / float64  浮点数
bool               布尔值
```

机器学习中：

- 标签通常是整数，因为类别索引需要整数；
- 特征、权重、梯度通常是浮点数；
- 图像原始像素可能是 `uint8`，预处理后应转为浮点数。

#### 转换类型

```python
x_float = x.astype(np.float64)
```

`astype` 默认返回新数组，不修改原数组。

#### 为什么先转浮点数

像素标准化：

```python
pixels = pixels.astype(np.float64) / 255.0
```

后续均值、梯度和参数更新都依赖小数。若把结果强行写回整数数组，小数部分可能丢失或触发类型转换错误。

#### 溢出

`uint8` 只能表示有限范围的整数。直接进行像素加法、平方等操作，可能溢出而得到意外结果。进行模型计算前转成浮点数更稳妥。

#### 浮点误差

计算机中的浮点数是近似表示：

```python
0.1 + 0.2 == 0.3  # 可能为 False
```

数组测试使用：

```python
np.testing.assert_allclose(actual, expected)
```

而整数、索引和标签使用：

```python
np.testing.assert_array_equal(actual, expected)
```

### 3. 最小代码骨架

```python
import numpy as np

pixels = np.array([0, 127, 255], dtype=np.uint8)

# TODO：创建浮点标准化结果，不修改 pixels
scaled = ...
```

### 4. 如何验证

```python
assert np.issubdtype(scaled.dtype, np.floating)
assert pixels.dtype == np.uint8
np.testing.assert_allclose(scaled, np.array([0.0, 127 / 255, 1.0]))
```

### 5. 常见错误

- 只检查值，不检查 `dtype`。
- 认为整数数组能自然保存梯度更新后的浮点参数。
- 对浮点数组使用严格相等测试。
- 为节省内存过早使用很低精度类型；学习阶段优先保证含义清晰和结果正确。

### 6. 完成后发什么检查

发转换代码、转换前后的 `dtype`，并说明原数组是否改变。

---

## 第 9 节：副本、视图与输入是否被修改

### 1. 当前目标

理解 NumPy 中“两个变量是否共享底层数据”，避免函数意外修改调用者的数组。

### 2. 需要理解的概念

你已经接触过 Python 的引用和浅拷贝。NumPy 还多了一个非常重要的概念：view（视图）。

#### 赋值：同一个对象

```python
a = np.array([1, 2, 3])
b = a
b[0] = 99
```

`a[0]` 也会变，因为 `a` 和 `b` 指向同一个数组对象。

#### 切片：通常是视图

```python
a = np.array([1, 2, 3, 4])
b = a[1:3]
b[0] = 99
```

此时 `a[1]` 通常也会变。`b` 是看向 `a` 部分数据的窗口，没有复制底层数据。

#### 显式副本

```python
b = a[1:3].copy()
```

修改 `b` 不再影响 `a`。

#### 高级索引通常产生副本

布尔索引和整数数组索引通常会产生副本，而基本切片通常产生视图。不要靠猜测来决定安全性；当函数明确要求“不修改输入”时，可以在关键位置显式复制，或让表达式创建新结果。

例如：

```python
processed = np.asarray(x, dtype=np.float64) / 255.0
```

除法表达式会创建结果数组，因此不会把归一化值原地写进输入。

原地操作需要特别警惕：

```python
x /= 255.0
x -= mean
x[mask] = 0
```

这些写法尝试直接修改 `x` 的数据。

### 3. 最小代码骨架

```python
import numpy as np

original = np.array([10, 20, 30, 40])
before = original.copy()

# TODO：取中间两个元素，修改结果，但不能改变 original
middle = ...
middle[0] = 999
```

### 4. 如何验证

```python
np.testing.assert_array_equal(original, before)
np.testing.assert_array_equal(middle, np.array([999, 30]))
```

调试共享关系时还可使用：

```python
np.shares_memory(original, middle)
```

### 5. 常见错误

- 认为切片一定是独立新数组。
- 在预处理函数中使用原地运算，导致训练数据被悄悄改写。
- 认为 `reshape` 一定复制数据；很多情况下它也可能返回视图。
- 为避免所有风险到处无脑 `.copy()`；复制会消耗额外内存，应在所有权边界清楚时使用。

### 6. 完成后发什么检查

发 TODO、断言结果以及 `np.shares_memory(original, middle)` 的输出。

---

## 第 10 节：拼接、堆叠与 batch

### 1. 当前目标

区分沿已有轴拼接和创建新轴堆叠，并能用索引切出 mini-batch。

### 2. 需要理解的概念

#### `concatenate`：沿已有轴连接

```python
a.shape == (2, 3)
b.shape == (1, 3)

c = np.concatenate([a, b], axis=0)
c.shape == (3, 3)
```

沿 `axis=0` 连接时，其他轴必须匹配。

#### `stack`：创建新轴

```python
a.shape == (3,)
b.shape == (3,)

rows = np.stack([a, b], axis=0)  # (2, 3)
cols = np.stack([a, b], axis=1)  # (3, 2)
```

简单记忆：

```text
concatenate：在已有方向上接长
stack：先增加一个新方向，再把数组放进去
```

#### mini-batch

设：

```text
X.shape == (N, D)
y.shape == (N,)
```

第 `start` 到 `end` 个样本：

```python
x_batch = x[start:end]
y_batch = y[start:end]
```

特征和标签必须使用同一组样本索引，才能保持一一对应。

若有打乱后的索引：

```python
indices = np.array([3, 0, 2, 1])
x_shuffled = x[indices]
y_shuffled = y[indices]
```

### 3. 最小代码骨架

```python
import numpy as np

x = np.arange(20).reshape(10, 2)
y = np.arange(10)
batch_size = 3

batches = []
for start in range(0, len(x), batch_size):
    end = start + batch_size
    # TODO：切出对应的 x_batch 和 y_batch
    # TODO：把二元组加入 batches
```

这里允许 Python 循环，因为循环处理的是 batch 边界，不是逐个元素计算。

### 4. 如何验证

```python
assert len(batches) == 4
assert batches[0][0].shape == (3, 2)
assert batches[-1][0].shape == (1, 2)
np.testing.assert_array_equal(batches[2][1], np.array([6, 7, 8]))
```

### 5. 常见错误

- 特征打乱了但标签没有使用同一组索引。
- 丢弃最后一个不足 `batch_size` 的 batch，除非题目明确要求。
- 混淆 `stack` 与 `concatenate`，得到多一个或少一个轴。
- 用 Python 循环逐个复制 batch 中的元素，没有利用切片。

### 6. 完成后发什么检查

发循环中的两行 TODO 和每个 batch 的 `(x_batch.shape, y_batch.shape)`。

---

## 第 11 节：随机数、打乱与可复现性

### 1. 当前目标

使用独立随机数生成器打乱样本，并让实验可以复现。

### 2. 需要理解的概念

机器学习会用随机数初始化参数、打乱数据、抽样等。学习代码中推荐：

```python
rng = np.random.default_rng(seed)
```

`seed` 是随机数生成器的初始状态来源。同一个 NumPy 版本和同一种调用顺序下，相同 seed 可以让实验重复得到同一序列，便于调试。

生成打乱索引：

```python
indices = rng.permutation(len(x))
```

再将同一索引用于特征和标签：

```python
x_shuffled = x[indices]
y_shuffled = y[indices]
```

不要分别打乱 `x` 和 `y`，否则样本与标签可能错位。

常见接口：

```python
rng.random((2, 3))             # [0, 1) 均匀分布
rng.normal(0.0, 1.0, (2, 3))  # 正态分布
rng.integers(0, 10, size=5)   # 随机整数
rng.permutation(5)             # 0 到 4 的随机排列
```

### 3. 最小代码骨架

```python
import numpy as np

x = np.array([[10], [20], [30], [40]])
y = np.array([1, 2, 3, 4])

rng = np.random.default_rng(42)

# TODO：生成索引并同步打乱 x、y
```

### 4. 如何验证

```python
# 不依赖具体排列，只验证配对关系没有破坏
np.testing.assert_array_equal(x_shuffled[:, 0] / 10, y_shuffled)

# 重新创建同 seed 的生成器，应得到同一组索引
rng_again = np.random.default_rng(42)
np.testing.assert_array_equal(indices, rng_again.permutation(len(x)))
```

### 5. 常见错误

- 对 `x`、`y` 分别调用随机排列。
- 每次需要随机数时都重新创建相同 seed 的生成器，导致每次都得到相同开头。
- 不设 seed，测试失败时难以复现。
- 依赖某一组“看起来随机”的具体数字，而不是验证更重要的性质。

### 6. 完成后发什么检查

发索引、打乱后的 `x` 和 `y`，以及两个断言的结果。

---

## 第 12 节：标签、one-hot 与准确率的 NumPy 写法

### 1. 当前目标

把已经用纯 Python 实现过的标签处理，翻译成基于数组索引和布尔运算的 NumPy 思路。

### 2. 需要理解的概念

#### one-hot

标签：

```python
labels = np.array([2, 0, 1])  # shape (N,)
```

若类别数为 $C=3$，one-hot 矩阵 $Y \in \{0,1\}^{N \times C}$ 满足：

$$Y_{ic} =
\begin{cases}
1, & c = y_i \\
0, & c \ne y_i
\end{cases}$$

符号对应：

| 数学符号 | 含义 | 代码对应 |
|---|---|---|
| $N$ | 标签数，也就是样本数 | `labels.shape[0]` |
| $C$ | 类别数 | `num_classes` |
| $y_i$ | 第 `i` 个样本的类别编号 | `labels[i]` |
| $Y$ | one-hot 矩阵 | `encoded`，shape `(N, C)` |

先创建全 0 数组：

```python
encoded = np.zeros((len(labels), num_classes), dtype=int)
```

然后需要把这些坐标设为 1：

```text
行坐标：0, 1, 2, ..., N-1
列坐标：labels 中记录的类别
```

你需要的函数是：

```python
np.arange(len(labels))
```

以及二维高级索引：

```python
array[行索引数组, 列索引数组]
```

这里故意不写完整赋值语句，留给你完成。

#### 从 logits 得到预测

```text
logits.shape == (N, C)
```

每个样本选择分数最大的类别，所以要沿类别轴求最大位置：

```python
predictions = logits.argmax(axis=1)
```

#### 准确率

$$\text{accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\hat{y}_i = y_i)$$

| 符号 | 含义 | 代码对应 |
|---|---|---|
| $N$ | 样本数 | `len(labels)` |
| $\hat{y}_i$ | 模型预测标签 | `predictions[i]` |
| $y_i$ | 真实标签 | `labels[i]` |
| $\mathbf{1}(条件)$ | 条件成立为 1，否则为 0 | 布尔比较结果 |

NumPy 中布尔数组求平均值时，`True` 视为 1，`False` 视为 0。因此需要的两个操作是：

```python
predictions == labels
np.mean(...)
```

仍然把最终表达式留给你完成。

### 3. 最小代码骨架

```python
import numpy as np

def one_hot(labels, num_classes):
    labels = np.asarray(labels)
    encoded = np.zeros((len(labels), num_classes), dtype=int)
    rows = np.arange(len(labels))

    # TODO：用 rows 和 labels 一次性写入所有 1
    return encoded


def accuracy_from_logits(logits, labels):
    # TODO 1：得到每个样本的预测类别
    # TODO 2：比较预测与真实标签并求平均
    return ...
```

### 4. 如何验证

```python
labels = np.array([2, 0, 1])
expected = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
])
np.testing.assert_array_equal(one_hot(labels, 3), expected)

logits = np.array([
    [0.1, 0.2, 0.9],
    [3.0, 1.0, 0.0],
    [0.5, 0.4, 0.1],
])
assert accuracy_from_logits(logits, labels) == 2 / 3
```

### 5. 常见错误

- 沿 `axis=0` 求 `argmax`，得到每个类别对应的样本，而非每个样本的类别。
- one-hot 的行列索引写反。
- 标签不是整数或超出 `[0, C)` 范围。
- logits 和 labels 的样本数量不同。

### 6. 完成后发什么检查

发两个函数的实现、输出 shape 和自测结果。

---

## 第 13 节：数组测试与 shape 调试法

### 1. 当前目标

形成一套定位 NumPy 错误的固定流程，而不是靠反复尝试转置。

### 2. 需要理解的概念

遇到 NumPy 问题时，按以下顺序检查：

```text
1. 每个轴在业务上代表什么？
2. 输入 shape 是什么？
3. 操作会消除、增加还是重排哪个轴？
4. 预期输出 shape 是什么？
5. dtype 是否适合当前运算？
6. 是否意外修改了输入？
7. 最后才检查具体数值。
```

推荐临时输出：

```python
print("x:", x.shape, x.dtype)
print("w:", w.shape, w.dtype)
print("logits:", logits.shape, logits.dtype)
```

推荐断言：

```python
assert x.ndim == 2
assert labels.ndim == 1
assert len(x) == len(labels)
assert x.shape[1] == w.shape[0]
```

数组比较：

```python
np.testing.assert_array_equal(actual_int, expected_int)
np.testing.assert_allclose(actual_float, expected_float)
```

`assert_allclose` 判断的是：误差是否在允许范围内，而不是二进制表示是否完全相同。

### 3. 最小代码骨架

为之后每道题保留这个调试模板：

```python
def some_numpy_function(x):
    x = np.asarray(x)

    # 输入契约
    if x.ndim != 2:
        return None

    # 核心计算
    result = ...

    # 开发阶段的 shape 检查
    assert result.shape[0] == x.shape[0]
    return result
```

这不是要求所有函数都返回 `None`。错误输入应该返回 `None`、抛出异常还是继续处理，取决于题目或项目 API 的约定。

### 4. 如何验证

至少准备三类测试：

```text
正常输入
最小或空输入
shape / dtype 不符合预期的输入
```

对“不修改输入”的函数：

```python
before = x.copy()
result = function(x)
np.testing.assert_array_equal(x, before)
```

### 5. 常见错误

- 只打印整个大数组，信息太多，却不打印 shape 和 dtype。
- 报 shape 错误后随手加 `.T`，没有解释轴的业务含义。
- 测试只覆盖一个正常样例。
- 把测试写成与实现完全相同的计算，导致两边可能犯同一个错。

### 6. 完成后发什么检查

发失败的完整异常、相关数组的 shape/dtype，以及你预期的输出 shape。不要只发最后一行报错。

---

## 第 14 节：进入现有 NumPy + ML 练习

### 1. 当前目标

把本教程知识映射到 `numpy-ml-foundations` 中的练习，而不是继续孤立地背 API。

### 2. 需要理解的概念

现有练习与教程章节的对应关系：

| 练习 | 主要知识 |
|---|---|
| P01 | 纯 Python 数据配对与 batch，作为 NumPy 前基线 |
| P02 | `shape`、`ndim`、`reshape`、`sum(axis=...)`、`argmax` |
| P03 | `dtype`、逐元素运算、`mean/std`、broadcasting、数据泄漏 |
| P04 | 增加维度、broadcasting、距离矩阵、`argmin` |
| P05 | 高级索引、one-hot、`argmax`、布尔均值、`np.add.at` |
| P06 | `default_rng`、`permutation`、高级索引、mini-batch |
| P07-P10 | 矩阵乘法、线性分类器、梯度与训练循环 |
| P11-P13 | 两层网络、激活函数、反向传播与训练 |

现在最合适的小目标是只做 P02，不要提前写后面的完整网络。

### 3. 最小代码或操作骨架

打开：

```text
numpy-ml-foundations/p02_image_batch_summary.py
```

你需要的工具只有：

```python
np.asarray(data)
array.ndim
array.shape
array.reshape(...)
array.sum(axis=...)
array.argmax(axis=...)
```

实现顺序：

```text
1. 转成 ndarray
2. 验证 ndim
3. 保存原 shape
4. 保留样本轴，展平每张图
5. 沿像素轴求和
6. 沿像素轴找最大值位置
7. 组装返回字典
```

不要在样本或像素维度上使用 Python 循环。

### 4. 如何验证

运行题目文档中的自测，并额外检查：

```python
assert summary["flat"].shape[0] == images.shape[0]
assert summary["pixel_sums"].shape == (images.shape[0],)
assert summary["brightest_indices"].shape == (images.shape[0],)
```

### 5. 常见错误

- `reshape(-1)` 把所有图片合并成一个向量。
- 求和时 axis 选错，得到每个像素跨图片的总和。
- `argmax()` 没有传 axis，只返回一个全局位置。
- 写循环逐张处理，绕过了本题要练的向量化。

### 6. 完成后发什么检查

发 `p02_image_batch_summary.py` 的函数实现和运行输出。我会只批改当前 P02，再继续 P03。

---

## NumPy 机器学习速查表

这张表用于回忆，不代替前面的 shape 推导。

| 目标 | 常见写法 | shape 变化 |
|---|---|---|
| 转数组 | `np.asarray(x)` | 取决于输入 |
| 看形状 | `x.shape` | 不变 |
| 看轴数 | `x.ndim` | 不变 |
| 转浮点 | `x.astype(np.float64)` | 不变 |
| 展平每个样本 | `x.reshape(len(x), -1)` | `(N, ...) -> (N, D)` |
| 增加中间轴 | `x[:, None, :]` | `(N, D) -> (N, 1, D)` |
| 转置二维数组 | `x.T` | `(N, D) -> (D, N)` |
| 每个样本求和 | `x.sum(axis=1)` | `(N, D) -> (N,)` |
| 每个特征求均值 | `x.mean(axis=0)` | `(N, D) -> (D,)` |
| 每个样本取最大类别 | `x.argmax(axis=1)` | `(N, C) -> (N,)` |
| 逐元素乘 | `x * y` | 广播后的 shape |
| 矩阵乘法 | `x @ w` | `(N, D) @ (D, C) -> (N, C)` |
| 条件筛选 | `x[x > 0]` | 通常变成一维 |
| 按索引重排 | `x[indices]` | 第一轴按索引重排 |
| 复制 | `x.copy()` | shape 不变，数据独立 |
| 随机排列 | `rng.permutation(n)` | `(n,)` |

---

## 最重要的五条习惯

1. 写 NumPy 代码前，先在纸上写每个数组的 shape。
2. 永远说明每个轴的业务含义，例如“样本轴、特征轴、类别轴”。
3. 遇到 `axis`，先说清楚要压缩哪个轴。
4. 区分逐元素乘法 `*` 和矩阵乘法 `@`。
5. 浮点数组用 `assert_allclose`，并测试函数是否意外修改输入。

---

## 官方参考资料

- [NumPy User Guide](https://numpy.org/doc/stable/user/)
- [NumPy absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Copies and views](https://numpy.org/doc/stable/user/basics.copies.html)
- [`numpy.matmul`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
- [`numpy.sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)

查文档时先关注四项：输入参数、返回值、shape 行为和示例。暂时不需要背完整 API。
