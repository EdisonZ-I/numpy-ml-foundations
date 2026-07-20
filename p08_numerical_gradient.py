# P08: numerical gradient checker
import numpy as np

def numerical_gradient(loss_function, parameter, epsilon=1e-5):
    result = np.zeros(parameter.shape)
    for i in np.ndindex(parameter.shape):
        para_plus = parameter.copy()
        para_minus = parameter.copy()
        para_plus[i] += epsilon
        para_minus[i] -= epsilon
        result[i] = (loss_function(para_plus) - loss_function(para_minus)) / (2 * epsilon)
    return result

if __name__ == "__main__":
    from test_helpers import check_allclose, check_array_equal
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