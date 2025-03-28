from typing import Any, Callable

import jax
import numpy as np
from jax import Array, lax
from jax import numpy as jnp
from jax.experimental import io_callback
from jax.typing import ArrayLike

import pintax
from pintax import Quantity, areg, quantity, unitify, ureg


@unitify
def test_scan():
    def inner(c, a):
        c0, c1 = c
        return (c0 + c1, a * areg.s), c0

    (ans, _), _ = lax.scan(inner, init=(0.0, 0.0), xs=jnp.arange(5) * 1.0 * areg.m)
    assert ans == 6 * areg.m * areg.s


@unitify
def test_callback():

    x = 5.0 * areg.m

    def debug_cb(x):
        assert x == 5.0 * areg.m

    jax.debug.callback(debug_cb, x)

    assert jax.pure_callback(lambda x: quantity(x).m + 1, x.aval, x) == 6.0

    x_np = None

    def io_cb(x):
        nonlocal x_np
        x_np = np.array(quantity(x).m)

    io_callback(io_cb, None, x)
    assert x_np == 5
