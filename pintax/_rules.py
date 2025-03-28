from __future__ import annotations

import jax
from jax import Array
from jax._src import core, pjit, traceback_util
from jax._src.core import Primitive
from jax._src.typing import ArrayLike
from jax.interpreters.partial_eval import (
    convert_invars_to_constvars,
)

from ._core import (
    PintaxError_forward,
    PintaxTypeError,
    Qt,
    Unit,
    UnitTracer,
    anyunit,
    dimensionless,
    quantity,
    rules_complex,
    with_unit_trace,
)
from ._utils import (
    cast_unchecked,
    check_unit,
    pp_join,
    pp_nested,
    pretty_print,
)
from ._utils import safe_zip as zip
from ._utils import (
    weakref_lru_cache_,
)

traceback_util.register_exclusion(__file__)


@weakref_lru_cache_
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
        raise e.ex_type(new_msg) from None
        # raise PintaxError_forward(ex_type=e.ex_type, msg=new_msg) from None


def unitify_jaxpr_(
    jaxpr: core.Jaxpr,
    in_units: tuple[Unit, ...],
    force_out_units: tuple[Unit | None, ...] | None = None,
) -> tuple[core.Jaxpr, tuple[Unit, ...]]:

    def inner(consts: tuple[ArrayLike], *args: Qt):
        with with_unit_trace() as trace:
            args_arr = [UnitTracer(trace, x) for x in args]
            out_bufs: list[ArrayLike] = core.eval_jaxpr(jaxpr, consts, *args_arr)
            ans = [trace.handle_primitive_arg(x) for x in out_bufs]
            if force_out_units is not None:
                ans = [
                    x if u is None else x._to(u) for x, u in zip(ans, force_out_units)
                ]
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
