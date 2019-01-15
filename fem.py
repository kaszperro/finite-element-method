from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def get_base_function(k):
    return lambda x: max((1 - abs(x * n - k)), 0)


def get_base_function_der(k):
    return lambda x: 0 if x < (k - 1) / n or x > (k + 1) / n else n if x < k / n else -n


def calc_b(u_der: Callable, v_der: Callable, u: Callable, v: Callable, start: float, end: float) -> float:
    def _first(x): return function_a(x) * u_der(x) * v_der(x)

    def _second(x): return function_b(x) * u_der(x) * v(x)

    def _third(x): return function_c(x) * u(x) * v(x)

    return -beta * u(0) * v(0) + quad(_first, start, end)[0] + quad(_second, start, end)[0] + quad(_third, start, end)[
        0]


def calc_l(v: Callable, start, end) -> float:
    return -gamma * v(0) + quad(lambda x: function_f(x) * v(x), start, end)[0]


def calc_b_in_matrix(i: int, j: int) -> float:
    start = max(0.0, (i - 1) / n)
    end = min(1.0, (j + 1) / n)

    return calc_b(
        get_base_function_der(i),
        get_base_function_der(j),
        get_base_function(i),
        get_base_function(j),
        start,
        end
    )


def get_u() -> Callable:
    def shift(x):
        return get_base_function(n)(x) * u1

    matrix = np.empty((n, n))

    for i in range(0, n):
        for j in range(0, n):
            matrix[i][j] = calc_b_in_matrix(j, i)

    right_matrix = np.empty(n)

    for i in range(0, n):
        right_matrix[i] = calc_l(get_base_function(i), max(0.0, (i - 1) / n), min(1.0, (i + 1) / n)) - calc_b(
            lambda x: get_base_function_der(n)(x) * u1,
            get_base_function_der(i),
            shift,
            get_base_function(i), max(0.0, (i - 1) / n), min(1.0, (i + 1) / n)
        )

        part_result = np.linalg.solve(matrix, right_matrix)

    return lambda x: shift(x) + sum(
        u * get_base_function(iu)(x) for iu, u in enumerate(part_result)
    )


if __name__ == '__main__':
    def function_a(x): return 0


    def function_b(x): return 1


    def function_c(x): return 0


    def function_f(x): return x


    beta = 0
    gamma = 0
    u1 = 1 / 2
    n = 100

    u_function = get_u()

    x_space = np.linspace(0, 1, 50)
    y_res = [u_function(x) for x in x_space]
    plt.plot(x_space, y_res)
    plt.show()
