from __future__ import annotations

from functools import partial

from jax import Array
from jax import numpy as jnp
from jax._src import ad_util, core, traceback_util
from jax._src.interpreters import ad, batching
from jax._src.traceback_util import api_boundary
from jax._src.typing import ArrayLike
from jax.interpreters import mlir

from ._core import (
    Qt,
    Unit,
    anyunit,
    dimensionless,
    quantity,
    rules_complex,
)
from ._utils import (
    cast_unchecked,
    dict_set,
)

traceback_util.register_exclusion(__file__)


def _ensure_arr(x: ArrayLike) -> Array:
    if not isinstance(x, Array):
        x = jnp.array(x)
        assert isinstance(x, Array)
        return x
    return x


def make_unit(x: ArrayLike, unit: Unit) -> Array:
    assert isinstance(unit, Unit)
    ans = make_unit_p.bind(_ensure_arr(x), unit=unit)
    assert isinstance(ans, Array)
    return ans


def value_and_unit(value: ArrayLike) -> tuple[Array, Array]:
    val, unit = value_and_unit_p.bind(_ensure_arr(value))
    assert isinstance(val, Array)
    assert isinstance(unit, Array)
    return val, unit


def convert_unit(value: ArrayLike, unit: ArrayLike) -> Array:
    val = convert_unit_p.bind(_ensure_arr(value), _ensure_arr(unit))
    assert isinstance(val, Array)
    return val


make_unit_p = core.Primitive("make_unit")


@make_unit_p.def_abstract_eval
def _(x: core.ShapedArray, /, *, unit: Unit):
    assert isinstance(unit, Unit)
    return x


@rules_complex(make_unit_p)
def _(x: Qt, /, *, unit: Unit):
    # assert x._unit in [dimensionless, anyunit]
    # return quantity(x._val, unit)

    from ._rules import mul_units

    return quantity(x._val, mul_units(x._unit, unit))


@dict_set(ad.primitive_jvps, make_unit_p)
def _(primals: tuple[ArrayLike], tangents: tuple[ArrayLike], /, *, unit: Unit):
    ans = make_unit_p.bind(*primals, unit=unit)
    (t,) = tangents
    return ans, t / make_unit(1, unit)


@make_unit_p.def_impl
def _(x: ArrayLike, /, *, unit: Unit):
    raise TypeError(f"trying to use {unit!r} outside of unitify")


@partial(mlir.register_lowering, make_unit_p)
@cast_unchecked[mlir.LoweringRule]()
def _(ctx, x: ArrayLike, /, *, unit: Unit):
    raise TypeError(f"trying to use {unit!r} outside of unitify")


convert_unit_p = core.Primitive("convert_unit")


@convert_unit_p.def_abstract_eval
def _(x: core.ShapedArray, y: core.ShapedArray, /):
    return x


@rules_complex(convert_unit_p)
def _(x: Qt, y: Qt, /) -> Qt:
    return x._to(y._unit)


# value_and_unit_p
value_and_unit_p = core.Primitive("value_and_unit")
value_and_unit_p.multiple_results = True


@value_and_unit_p.def_abstract_eval
def _(arg: core.ShapedArray):
    return arg, core.ShapedArray(shape=(), dtype=jnp.int32)


@rules_complex(value_and_unit_p)
def _(x: Qt):
    return quantity(x._val, dimensionless), quantity(1, x._unit)


@dict_set(ad.primitive_jvps, value_and_unit_p)
def _(primals: tuple[ArrayLike], tangents: tuple[ArrayLike]):
    val, unit = value_and_unit_p.bind(*primals)
    (t,) = tangents
    return (val, unit), (
        t / unit,
        ad_util.Zero(core.ShapedArray(shape=(), dtype=jnp.int32)),
    )


@dict_set(batching.primitive_batchers, value_and_unit_p)
def _(batched_args: tuple[ArrayLike], batch_dims: tuple[int]):
    (bd,) = batch_dims
    return value_and_unit_p.bind(*batched_args), (bd, batching.not_mapped)


@value_and_unit_p.def_impl
def _(*_, **__):
    raise TypeError(f"trying to use {value_and_unit} outside of unitify")


@partial(mlir.register_lowering, value_and_unit_p)
def _(*_, **__):
    raise TypeError(f"trying to use {value_and_unit} outside of unitify")
