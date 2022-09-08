# -- encoding:utf-8 --
"""
Create on 19/3/10
"""

import numpy as np


def entropy(p):
    return np.sum([-t * np.log2(t) for t in p])


print(entropy([0.25, 0.25, 0.25, 0.25]))
print(entropy([0.5, 0.5]))
print(entropy([0.65, 0.25, 0.05, 0.05]))
print(entropy([0.97, 0.01, 0.01, 0.01]))
print(entropy([0.5, 0.25, 0.25]))
print(entropy([0.7, 0.3]))
print(entropy([5.0 / 8, 3.0 / 8]))
print(entropy([0.4, 0.6]))
