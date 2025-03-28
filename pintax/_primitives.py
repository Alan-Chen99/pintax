from __future__ import annotations

from functools import partial

from jax import Array
from jax import numpy as jnp
from jax._src import ad_util, core, traceback_util
from jax._src.interpreters import ad, batching
from jax._src.typing import ArrayLike
from jax.interpreters import mlir

from ._core import (
    Qt,
    Unit,
    dimensionless,
    mul_units,
    quantity,
    rules_complex,
)
from ._utils import (
    cast_unchecked,
    check_arraylike,
    dict_set,
)

traceback_util.register_exclusion(__file__)


def prim_make_unit(x: ArrayLike, unit: Unit) -> Array:
    assert isinstance(unit, Unit)
    ans = make_unit_p.bind(check_arraylike(x), unit=unit)
    assert isinstance(ans, Array)
    return ans


def prim_value_and_unit(value: ArrayLike) -> tuple[Array, Array]:
    val, unit = value_and_unit_p.bind(check_arraylike(value))
    assert isinstance(val, Array)
    assert isinstance(unit, Array)
    return val, unit


def prim_convert_unit(value: ArrayLike, unit: ArrayLike) -> Array:
    """convert value to the unit of unit; return compares equal to value."""
    val = convert_unit_p.bind(check_arraylike(value), check_arraylike(unit))
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

    return quantity(x._val, mul_units(x._unit, unit))


@dict_set(ad.primitive_jvps, make_unit_p)
def _(primals: tuple[ArrayLike], tangents: tuple[ArrayLike], /, *, unit: Unit):
    ans = make_unit_p.bind(*primals, unit=unit)
    (t,) = tangents
    return ans, t / prim_make_unit(1, unit)


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


@dict_set(batching.primitive_batchers, convert_unit_p)
def _(
    batched_args: tuple[ArrayLike, ArrayLike], batch_dims: tuple[int | None, int | None]
):
    x, y = batched_args
    x_b, _ = batch_dims
    return convert_unit_p.bind(x, y), x_b


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
    raise TypeError(f"trying to use {prim_value_and_unit} outside of unitify")


@partial(mlir.register_lowering, value_and_unit_p)
def _(*_, **__):
    raise TypeError(f"trying to use {prim_value_and_unit} outside of unitify")
