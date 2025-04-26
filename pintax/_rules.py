from __future__ import annotations

from typing import Callable

import jax
from jax import Array
from jax._src import core, pjit, traceback_util
from jax._src.typing import ArrayLike
from jax.interpreters.partial_eval import (
    convert_invars_to_constvars,
)

from ._core import (
    PintaxError_forward,
    PintaxTypeError,
    Qt,
    Unit,
    UnitTrace,
    UnitTracer,
    anyunit,
    as_multiplicative,
    assert_multiplicative,
    dimensionless,
    div_units,
    is_multiplicative,
    mul_units,
    quantity,
    rules_complex,
    with_unit_trace,
)
from ._utils import (
    cast_unchecked,
    pp_join,
    pp_nested,
    pretty_print,
)
from ._utils import safe_zip as zip
from ._utils import (
    weakref_lru_cache_,
)

traceback_util.register_exclusion(__file__)


def unitify_jaxpr(
    jaxpr: core.Jaxpr,
    in_units: tuple[Unit, ...],
    force_out_units: tuple[Unit | None, ...] | None = None,
) -> tuple[core.Jaxpr, tuple[Unit, ...]]:
    try:
        return unitify_jaxpr_(jaxpr, in_units, force_out_units)
    except PintaxError_forward as e:
        new_msg = pp_join(
            pp_nested("failed to unitify jaxpr:", e.msg),
            pp_nested("in_units:", pretty_print(in_units)),
            pp_nested("force_out_units:", pretty_print(force_out_units)),
            pp_nested("jaxpr:", pretty_print(jaxpr)),
        )
        raise e.unwrap(new_msg) from None
        # raise PintaxError_forward(ex_type=e.ex_type, msg=new_msg) from None


@weakref_lru_cache_
def unitify_jaxpr_(
    jaxpr: core.Jaxpr,
    in_units: tuple[Unit, ...],
    force_out_units: tuple[Unit | None, ...] | None = None,
) -> tuple[core.Jaxpr, tuple[Unit, ...]]:

    @with_unit_trace
    def inner(trace: UnitTrace, consts: tuple[ArrayLike], *args: Qt):
        args_arr = [UnitTracer(trace, x) for x in args]
        out_bufs: list[ArrayLike] = core.eval_jaxpr(jaxpr, consts, *args_arr)
        ans = [trace.handle_primitive_arg(x) for x in out_bufs]
        if force_out_units is not None:
            ans = [x if u is None else x._to(u) for x, u in zip(ans, force_out_units)]
        return ans

    out_jaxpr, out_shapes = jax.make_jaxpr(inner, return_shape=True)(
        tuple(x.aval for x in jaxpr.constvars),
        *[
            quantity(cast_unchecked[Array]()(x.aval), u)
            for u, x in zip(in_units, jaxpr.invars)
        ],
    )
    assert len(out_jaxpr.consts) == 0
    out_jaxpr_: core.Jaxpr = convert_invars_to_constvars(
        out_jaxpr.jaxpr, len(jaxpr.constvars)
    )
    assert len(out_jaxpr_.constvars) == len(jaxpr.constvars)

    def inner2(x):
        assert isinstance(x, Qt)
        return x._unit

    return out_jaxpr_, tuple(inner2(x) for x in out_shapes)


def sync_dims(*args: Qt) -> tuple[Unit, tuple[Qt, ...]]:
    dtype = args[0].dtype
    for x in args:
        if x.dtype != dtype:
            raise PintaxError_forward(
                ex_type=PintaxTypeError,
                msg=f"dtype mismatch: {dtype} and {x.dtype}",
            )
    nonzero = [x._unit for x in args if x._unit != anyunit]
    if len(nonzero) == 0:
        unit = anyunit
    else:
        unit = nonzero[0]
    return unit, tuple(x._to(unit) for x in args)


sync_dims_binop_t = Callable[[Qt, Qt], tuple[tuple[ArrayLike, ArrayLike], Unit]]


def sync_dims_binop_impl[F: sync_dims_binop_t](f: F) -> F:
    return f


@sync_dims_binop_impl
def sync_dims_for_concat(x: Qt, y: Qt):
    u, (x, y) = sync_dims(x, y)
    return (x._val, y._val), u


@sync_dims_binop_impl
def sync_dims_for_add(x: Qt, y: Qt):
    if x._unit == anyunit:
        return (x._val, y._val), y._unit
    if y._unit == anyunit:
        return (x._val, y._val), x._unit

    if is_multiplicative(y._unit):
        return (x._val, y._to(as_multiplicative(x._unit))._val), x._unit
    if is_multiplicative(x._unit):
        return (x._to(as_multiplicative(y._unit))._val, y._val), y._unit

    assert_multiplicative(y)
    assert False


@sync_dims_binop_impl
def sync_dims_for_scatter_add(x: Qt, y: Qt):
    assert_multiplicative(y)
    return sync_dims_for_add(x, y)


@sync_dims_binop_impl
def sync_dims_for_sub(x: Qt, y: Qt):
    if is_multiplicative(x._unit):
        assert_multiplicative(y)
        return sync_dims_for_concat(x, y)

    if x._unit == anyunit:
        return (x._val, y._val), y._unit
    if y._unit == anyunit:
        return (x._val, y._val), x._unit

    if is_multiplicative(y._unit):
        return (x._val, y._to(as_multiplicative(x._unit))._val), x._unit

    return sync_dims_for_concat(x, y)


@sync_dims_binop_impl
def sync_dims_for_mul(x: Qt, y: Qt):
    assert_multiplicative(x)
    assert_multiplicative(y)
    return (x._val, y._val), mul_units(x._unit, y._unit)


@sync_dims_binop_impl
def sync_dims_for_div(x: Qt, y: Qt):
    assert_multiplicative(x)
    assert_multiplicative(y)
    return (x._val, y._val), div_units(x._unit, y._unit)


def dimensionless_or_err(x: Unit):
    if x == dimensionless or x == anyunit:
        return
    raise PintaxError_forward(
        ex_type=PintaxTypeError,
        msg=f"expected dimensionless, got {x}",
    )


@rules_complex(pjit.pjit_p)
def _(
    *args: Qt,
    jaxpr: core.ClosedJaxpr,
    **kwargs,
):
    assert isinstance(jaxpr, core.ClosedJaxpr)

    for x in jaxpr.consts:
        assert not isinstance(x, core.Tracer)

    out_jaxpr, out_units = unitify_jaxpr(jaxpr.jaxpr, tuple(x._unit for x in args))

    out_vals: tuple[ArrayLike] = pjit.pjit_p.bind(
        *(x._val for x in args),
        jaxpr=core.ClosedJaxpr(out_jaxpr, jaxpr.consts),
        **kwargs,
    )

    return tuple(quantity(x, u) for x, u in zip(out_vals, out_units))
