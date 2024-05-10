import numpy as np
from typing import Tuple

from ...Population import Population


"""
用于生成交叉算子父代向量, 随机选择一半为p1, 另一半为p2.
"""


def random_selection(pop: Population) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回父代索引向量，如果种群大小不是偶数就随机去掉一个个体
    :param pop:
    :return: 返回索引向量
    """
    half = pop.pop_size // 2
    p = np.arange(0, pop.pop_size)
    np.random.shuffle(p)
    p1 = p[0:half]
    p2 = p[half:2*half]
    return p1, p2
