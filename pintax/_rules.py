from __future__ import annotations

import functools
from typing import Callable, Sequence

import jax
import numpy as np
from jax import Array, lax
from jax._src import ad_util, core, pjit, traceback_util
from jax._src.checkify import check_p
from jax._src.core import Primitive
from jax._src.debugging import debug_callback_p
from jax._src.typing import ArrayLike

from pintax._primitives import make_unit

from ._core import (
    PintaxError_forward,
    PintaxNotImplementedError,
    PintaxTypeError,
    Qt,
    Unit,
    UnitTracer,
    anyunit,
    dimensionless,
    quantity,
    rules,
    rules_complex,
    with_unit_trace,
)
from ._utils import cast_unchecked, check_unit, pp_join
from ._utils import safe_zip as zip
from ._utils import weakref_lru_cache_

traceback_util.register_exclusion(__file__)


@weakref_lru_cache_
def unitify_jaxpr(
    jaxpr: core.Jaxpr,
    in_units: tuple[Unit, ...],
    force_out_units: tuple[Unit | None, ...] | None = None,
) -> tuple[core.Jaxpr, tuple[Unit, ...]]:

    def inner(*args: Qt):
        with with_unit_trace() as trace:
            args_arr = [UnitTracer(trace, x) for x in args]
            out_bufs: list[ArrayLike] = core.eval_jaxpr(jaxpr, [], *args_arr)
            ans = [trace.ensure_quantity(x) for x in out_bufs]
            if force_out_units is not None:
                ans = [
                    x if u is None else x._to(u) for x, u in zip(ans, force_out_units)
                ]
            return ans

    out_jaxpr, out_shapes = jax.make_jaxpr(inner, return_shape=True)(
        *[
            quantity(cast_unchecked[Array]()(x.aval), u)
            for u, x in zip(in_units, jaxpr.invars)
        ]
    )
    assert len(out_jaxpr.consts) == 0

    def inner2(x):
        assert isinstance(x, Qt)
        return x._unit

    return out_jaxpr.jaxpr, tuple(inner2(x) for x in out_shapes)


_unary_linear = [
    lax.abs_p,
    lax.broadcast_in_dim_p,
    lax.copy_p,
    lax.imag_p,
    lax.neg_p,
    lax.real_p,
    lax.reduce_max_p,
    lax.reduce_sum_p,
    lax.reshape_p,
    lax.slice_p,
    lax.squeeze_p,
    lax.stop_gradient_p,
    lax.transpose_p,
]

# ordered as they are in jax/_src/lax/lax.py
_nounit_ops = [
    #
    lax.exp_p,
    lax.exp2_p,
    lax.log_p,
    lax.expm1_p,
    lax.log1p_p,
    lax.tanh_p,
    lax.logistic_p,
    #
    lax.sin_p,
    lax.cos_p,
    lax.tan_p,
    lax.asin_p,
    lax.acos_p,
    lax.atan_p,
    lax.atan2_p,
    lax.sinh_p,
    lax.cosh_p,
    lax.asinh_p,
    lax.acosh_p,
    lax.atanh_p,
    ##
    lax.pow_p,  # maybe special case
    lax.iota_p,
    lax.floor_p,  # TODO?
    ##
    lax.and_p,
    lax.reduce_and_p,
    lax.not_p,
    lax.or_p,
    ##
    check_p,
]


_mul_ops = [
    lax.dot_general_p,
    lax.mul_p,
]

_sametype_ops = [
    ad_util.add_any_p,
    lax.add_p,
    lax.concatenate_p,
    lax.max_p,
    lax.pad_p,
    lax.sub_p,
]

_sametype_nounit_ret = [
    lax.eq_p,
    lax.ge_p,
    lax.gt_p,
    lax.le_p,
    lax.lt_p,
    lax.ne_p,
]


@rules(lax.convert_element_type_p)
def _(x: Unit, new_dtype: np.dtype, **_):
    if x in [dimensionless, anyunit]:
        return x
    assert isinstance(new_dtype, np.dtype)
    if np.issubdtype(new_dtype, np.integer):
        raise PintaxError_forward(
            ex_type=PintaxNotImplementedError,
            msg="pintax for integers are not yet supported",
        )
    return x


@rules.many(_unary_linear)
def _(prim: Primitive):
    def func(one_arg: Unit, **kwargs):
        return one_arg

    return func


@rules_complex.many(_nounit_ops)
def _(prim: Primitive):
    def func(*args: Qt, **kwargs) -> Qt | list[Qt]:
        for x in args:
            if (
                x._unit != anyunit
                and x._unit.dimensionality != dimensionless.dimensionality
            ):
                raise PintaxError_forward(
                    ex_type=PintaxTypeError,
                    msg=f"expected dimensionless, got {x._unit}",
                )
        ans = prim.bind(*[x._to(dimensionless)._val for x in args], **kwargs)
        if prim.multiple_results:
            return [quantity(x, dimensionless) for x in ans]
        else:
            return quantity(ans, dimensionless)

    return func


def mul_units(x: Unit, y: Unit):
    if x == anyunit:
        x = dimensionless
    if y == anyunit:
        y = dimensionless
    return check_unit(x * y)


def div_units(x: Unit, y: Unit):
    if x == anyunit:
        x = dimensionless
    if y == anyunit:
        y = dimensionless
    return check_unit(x / y)


@rules.many(_mul_ops)
def _(prim: Primitive):
    def func(x: Unit, y: Unit, **kwargs):
        return mul_units(x, y)

    return func


@rules(lax.div_p)
def _(x: Unit, y: Unit):
    return div_units(x, y)


def sync_dims(prim: Primitive, args: tuple[Qt, ...]) -> tuple[Unit, tuple[Qt, ...]]:
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
        dm = unit.dimensionality
        for x in nonzero:
            if x.dimensionality != dm:
                msg = "\n".join(f"{x}" for x in args)
                raise PintaxError_forward(
                    ex_type=PintaxTypeError,
                    msg=pp_join(
                        "dimensionality mismatch:",
                        str(dm),
                        "and",
                        str(x.dimensionality),
                    ),
                )
    return unit, tuple(x._to(unit) for x in args)


@rules_complex.many(_sametype_ops)
def _(prim: Primitive):
    assert not prim.multiple_results

    def func(*args: Qt, **kwargs):
        u, args_ = sync_dims(prim, args)
        ans = prim.bind(*(x._val for x in args_), **kwargs)
        return quantity(ans, u)

    return func


@rules_complex.many(_sametype_nounit_ret)
def _(prim: Primitive):
    assert not prim.multiple_results

    def func(*args: Qt, **kwargs):
        _, args_ = sync_dims(prim, args)
        ans = prim.bind(*(x._val for x in args_), **kwargs)
        return quantity(ans, dimensionless)

    return func


@rules_complex(pjit.pjit_p)
def _(
    *args: Qt,
    jaxpr: core.ClosedJaxpr,
    **kwargs,
):
    assert isinstance(jaxpr, core.ClosedJaxpr)

    if len(jaxpr.jaxpr.constvars) > 0:
        raise NotImplementedError()

    out_jaxpr, out_units = unitify_jaxpr(
        jaxpr.jaxpr,
        tuple(x._unit for x in args),
    )

    out_vals: tuple[ArrayLike] = pjit.pjit_p.bind(
        *(x._val for x in args),
        jaxpr=core.ClosedJaxpr(out_jaxpr, jaxpr.consts),
        **kwargs,
    )

    return tuple(quantity(x, u) for x, u in zip(out_vals, out_units))


@rules(lax.integer_pow_p)
def _(x: Unit, y: int):
    if x == anyunit:
        assert y > 0
        return anyunit
    return check_unit(x**y)


@rules(lax.sqrt_p)
def _(x: Unit):
    if x == anyunit:
        return anyunit
    return check_unit(x**0.5)


@rules(lax.split_p)
def _(x: Unit, sizes: Sequence[int], **_):
    return tuple(x for _ in sizes)


@rules(lax.linalg.eig_p)
def _(
    x: Unit,
    compute_left_eigenvectors: bool,
    compute_right_eigenvectors: bool,
    **_,
):
    output = [x]
    if compute_left_eigenvectors:
        output.append(dimensionless)
    if compute_right_eigenvectors:
        output.append(dimensionless)
    return tuple(output)


@rules(lax.linalg.eigh_p)
def _(x: Unit, **_):
    return (dimensionless, x)


@rules(lax.argmin_p)
def _(x: Unit, **_):
    return dimensionless


@rules(lax.argmax_p)
def _(x: Unit, **_):
    return dimensionless


@rules(lax.reduce_min_p)
def _(x: Unit, **_):
    return x


@rules(lax.dynamic_slice_p)
def _(operand: Unit, *starts_and_dyn_sizes: Unit, **_):
    for x in starts_and_dyn_sizes:
        assert x == dimensionless
    return operand


@rules(lax.gather_p)
def _(operand: Unit, indices: Unit, **_):
    assert indices == dimensionless
    return operand


@rules(lax.linalg.svd_p)
def _(operand: Unit, compute_uv: bool, **_):
    if compute_uv:
        return (operand, dimensionless, dimensionless)
    else:
        return (operand,)


@rules_complex(lax.scatter_add_p)
def _(operand: Qt, scatter_indices: Qt, updates: Qt, **kwargs):
    assert scatter_indices._unit == dimensionless
    u, (operand, updates) = sync_dims(lax.scatter_add_p, (operand, updates))
    ans = lax.scatter_add_p.bind(
        operand._val, scatter_indices._val, updates._val, **kwargs
    )
    return quantity(ans, u)


@rules_complex(lax.select_n_p)
def _(which: Qt, *cases: Qt):
    assert which._unit in [dimensionless, anyunit]
    u, cases = sync_dims(lax.select_n_p, cases)
    ans = lax.select_n_p.bind(which._val, *(c._val for c in cases))
    return quantity(ans, u)


@weakref_lru_cache_
def make_debug_callback(callback: Callable, units: Sequence[Unit]):
    @functools.wraps(callback)
    def inner(*args: ArrayLike):
        with with_unit_trace() as trace:
            return callback(
                *[
                    x if u == dimensionless else make_unit(x, u)
                    for x, u in zip(args, units)
                ]
            )

    return inner


@rules_complex(debug_callback_p)
def _(*args: Qt, callback: Callable, **kwargs):
    new_callback = make_debug_callback(callback, tuple(x._unit for x in args))
    () = debug_callback_p.bind(*[x._val for x in args], callback=new_callback, **kwargs)
    return ()


@rules(debug_callback_p)
def _(*args: Unit, **_):
    return ()


@rules_complex(lax.scan_p)
def _(
    *args: Qt,
    jaxpr: core.ClosedJaxpr,
    num_carry: int,
    num_consts: int,
    **kwargs,
):
    assert len(jaxpr.consts) == 0

    # jaxpr: (n_consts, n_carry, n_xs) -> (n_carry, n_ys)

    num_ys = len(jaxpr.out_avals) - num_carry

    consts = args[:num_consts]
    carry = args[num_consts : num_consts + num_carry]
    xs = args[num_consts + num_carry :]

    consts_units = tuple(x._unit for x in consts)
    init_carry_units = tuple(x._unit for x in carry)
    xs_units = tuple(x._unit for x in xs)

    def get_carry_units() -> tuple[Unit, ...]:
        carry_units = init_carry_units
        for _ in range(1000):
            _, out_units = unitify_jaxpr(
                jaxpr.jaxpr,
                consts_units + carry_units + xs_units,
            )
            new_units = out_units[:num_carry]
            if not any(
                x == anyunit and y != anyunit for x, y in zip(carry_units, new_units)
            ):
                return new_units
            carry_units = new_units

        raise RuntimeError("infinite loop")

    carry_units = get_carry_units()

    carry = [x._to(u) for x, u in zip(carry, carry_units)]

    out_jaxpr, out_units = unitify_jaxpr(
        jaxpr.jaxpr,
        consts_units + carry_units + xs_units,
        force_out_units=tuple((*carry_units, *(None for _ in range(num_ys)))),
    )

    out_vals: tuple[ArrayLike] = lax.scan_p.bind(
        *(x._val for x in consts),
        *(x._val for x in carry),
        *(x._val for x in xs),
        jaxpr=core.ClosedJaxpr(out_jaxpr, jaxpr.consts),
        num_carry=num_carry,
        num_consts=num_consts,
        **kwargs,
    )
    return tuple(quantity(x, u) for x, u in zip(out_vals, out_units))


@rules_complex(lax.cond_p)
def _(which: Qt, *args: Qt, branches: tuple[core.ClosedJaxpr, ...]):
    assert which._unit == dimensionless

    for x in branches:
        assert len(x.consts) == 0

    out_units = [
        unitify_jaxpr(jaxpr.jaxpr, tuple(x._unit for x in args))[1]
        for jaxpr in branches
    ]

    out_units_cast_to = tuple(
        sync_dims(lax.cond_p, tuple(quantity(1.0, z) for z in x))[0]
        for x in zip(*out_units)
    )

    transformed_jaxprs = tuple(
        core.ClosedJaxpr(
            unitify_jaxpr(
                jaxpr.jaxpr,
                tuple(x._unit for x in args),
                force_out_units=out_units_cast_to,
            )[0],
            [],
        )
        for jaxpr in branches
    )

    out_bufs: list[ArrayLike] = lax.cond_p.bind(
        which._val, *(x._val for x in args), branches=transformed_jaxprs
    )
    return tuple(quantity(x, u) for x, u in zip(out_bufs, out_units_cast_to))
