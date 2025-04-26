from typing import Callable

import numpy as np
from jax import Array
from jax import numpy as jnp

from pintax import Quantity, areg, unitify, ureg


def check_one[*T](f: Callable[[*T], Quantity], *args: *T):
    assert (f(*args) == unitify(f)(*args)).all()


def checktype[T](tp: type[T]) -> Callable[[T], None]:
    def inner(val: T):
        assert isinstance(val, tp)

    return inner


def test_unitify_ret():
    checktype(Quantity)(unitify(lambda: jnp.array(0.0))())
    assert isinstance(unitify(lambda: jnp.array(0.0), unwrap_outs=True)(), Array)
    assert isinstance(unitify(lambda: areg.m, unwrap_outs=True)(), Quantity)
    checktype(Array)(unitify(lambda: jnp.array(0.0), force_dimensionless_outs=True)())


def test_arith():
    for f in [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x + 2 * y,
        lambda x, y: x - 2 * y,
        lambda x, y: x * y,
        lambda x, y: x / y,
        lambda x, y: x**2 + y**2,
    ]:
        check_one(f, 3.0 * ureg.m, 5 * ureg.m)
        check_one(f, 3.0 * ureg.inch, 5 * ureg.m)
        check_one(f, 3.0 * ureg.inch, ureg.m)

    for f in [
        lambda x, y: x * y,
        lambda x, y: x / y,
    ]:
        check_one(f, 3.0 * ureg.m, 5 * ureg.s)
        check_one(f, 3.0 * ureg.inch, 5 * ureg.s)
        check_one(f, 3.0 * ureg.inch, ureg.s)


def test_methods():
    q = np.arange(10) * ureg.m
    assert (q.reshape(2, 5).flatten() == q).all()
