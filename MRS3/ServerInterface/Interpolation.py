# replace s = (x - a) / (b - a)
# f(x) = (x - a) / (b - a)
# f(x) = 3x² - 2x³
# f(x) = 6x⁵ - 15x⁴ + 10x³
# f(x) = (1 - cos(πx)) / 2

import math
from multipledispatch import dispatch
from numbers import Number
import numpy as np

@dispatch(Number)
def linear(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return x

@dispatch(Number, Number, Number)
def linear(x, start, end):
    a = (x - start) / (end - start)
    return linear(a)

@dispatch(Number)
def hermit_3(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return 3 * x**2 - 2 * x**3

@dispatch(Number, Number, Number)
def hermit_3(x, start, end):
    a = (x - start) / (end - start)
    return hermit_3(a)

@dispatch(Number)
def hermit_5(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return 6 * x**5 - 15 * x**4 + 10 * x**3

@dispatch(Number, Number, Number)
def hermit_5(x, start, end):
    a = (x - start) / (end - start)
    return hermit_5(a)

@dispatch(Number)
def sinusoidal(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return (1 - math.cos(math.pi * x)) / 2

@dispatch(Number, Number, Number)
def sinusoidal(x, start, end):
    a = (x - start) / (end - start)
    return sinusoidal(a)

@dispatch(Number)
def unit_step(x):
    return 0 if x < 0 else 1

@dispatch(Number, Number, Number)
def unit_step(x, start, end):
    a = (x - start) / (end - start)
    return unit_step(a)

# TODO: cupy, numba 활용해서 ndarray 대상 작업(np_sth) gpu 가속
np_linear = np.vectorize(linear)
np_hermit_3 = np.vectorize(hermit_3)
np_hermit_5 = np.vectorize(hermit_5)
np_sinusoidal = np.vectorize(sinusoidal)
np_unit_step = np.vectorize(unit_step)

def step(a, b):
    def step_func(x):
        return 0 if x < b else a
    return step_func

# np.clip


