from __future__ import annotations

from typing import Any, Callable

import jax
import numpy as np
import pint
import pytest
from jax import Array
from jax import numpy as jnp
from jax import tree_util as jtu
from jax._src.traceback_util import api_boundary
from jax.typing import ArrayLike
from pytest_subtests import SubTests

from pintax import areg, sync_units, unitify
from pintax._core import PintaxDimensionalityError
from pintax._utils import cast_unchecked_, tree_map
from pintax.unstable import UnitTrace, anyunit, pint_registry, with_unit_trace

g = pint_registry()


def _pint_to_jax(q: Any) -> Array:
    if isinstance(q, pint.Quantity):
        return jnp.array(q.m) * areg(str(q.units))
    return jnp.array(q)


def rand(*shape: int) -> np.ndarray:
    return np.random.normal(0, 1, shape)


def check_eq(x: ArrayLike, y: ArrayLike):
    (x, y), _ = sync_units((x, y))
    assert jnp.allclose(x, y)


def run_one[**P](
    jax_fn: Callable[P, Any], np_fn: Callable[P, Any], ex: type[Exception] | None = None
):
    def inner(*args: P.args, **kwargs: P.kwargs):
        @api_boundary
        @with_unit_trace
        def do_test1(trace: UnitTrace):
            __tracebackhide__ = True
            jax_args, jax_kwargs = tree_map(_pint_to_jax, (args, kwargs))
            jax_ans = jax_fn(*jax_args, **jax_kwargs)

            assert len(jtu.tree_leaves(jax_ans)) > 0

            np_ans = tree_map(_pint_to_jax, np_fn(*args, **kwargs))

            tree_map(check_eq, np_ans, jax_ans)

        def do_test2():
            __tracebackhide__ = True
            if ex is None:
                do_test1()
            else:
                with pytest.raises(ex):
                    do_test1()

        __tracebackhide__ = True
        with jax.disable_jit():
            do_test2()

        do_test2()

    return inner


class testbuilder[**P]:
    def __init__(
        self,
        *functions: tuple[Callable[P, Any], Callable[P, Any]],
    ):
        self.functions = functions
        self.cases: list[tuple[tuple, dict, type[Exception] | None]] = []
        self.did_build = False

    def add(self, *args: P.args, **kwargs: P.kwargs):
        self.cases.append((args, kwargs, None))
        return self

    def raises(self, ex: type[Exception], *args: P.args, **kwargs: P.kwargs):
        self.cases.append((args, kwargs, ex))
        return self

    def build(self):
        self.did_build = True

        def inner(subtests: SubTests):
            for jax_fn, np_fn in self.functions:
                for args, kwargs, ex in self.cases:
                    with subtests.test(msg=str(jax_fn)):
                        run_one(jax_fn, np_fn, ex)(*args, **kwargs)

        return inner

    def __del__(self):
        if not self.did_build:
            raise RuntimeError()


test_add_sub = (
    testbuilder(
        (jnp.add, np.add),
        (jnp.subtract, np.subtract),
    )
    .add(rand(3, 5) * g.m, rand(3, 5) * g.mm)
    .raises(PintaxDimensionalityError, rand(3, 5) * g.m, rand(3, 5) * g.s)
    .add(0.0, rand(3, 5) * g.mm)
    .add(rand(3, 5) * g.mm, 0.0)
    .add(rand(3, 5), 2.0)
    .build()
)


test_mul_div = (
    testbuilder(
        (jnp.multiply, np.multiply),
        (jnp.divide, np.divide),
    )
    .add(rand(3, 5) * g.m, rand(3, 5) * g.mm)
    .add(rand(3, 5) * g.m, rand(3, 5) * g.s)
    .add(0.0, rand(3, 5) * g.mm)
    .add(rand(3, 5), 2.0)
    .build()
)


test_concat = (
    testbuilder(
        (jnp.concatenate, cast_unchecked_(np.concatenate)),
    )
    .add([rand(3, 5) * g.m, rand(3, 5) * g.mm], axis=1)
    .add(rand(2, 3, 5) * g.m)
    .build()
)
