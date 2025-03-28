from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from jax import lax
from jax._src import ad_util, traceback_util
from jax._src.checkify import check_p
from jax._src.core import Primitive

from ._core import (
    PintaxError_forward,
    PintaxNotImplementedError,
    PintaxTypeError,
    Qt,
    Unit,
    anyunit,
    check_unit,
    dimensionless,
    div_units,
    mul_units,
    pow_unit,
    quantity,
    rules,
    rules_complex,
)
from ._rules import dimensionless_or_err, sync_dims

traceback_util.register_exclusion(__file__)


_unary_linear: list[Primitive] = [
    lax.abs_p,
    lax.broadcast_in_dim_p,
    lax.ceil_p,
    lax.conj_p,
    lax.copy_p,
    lax.cumsum_p,
    lax.floor_p,
    lax.imag_p,
    lax.neg_p,
    lax.real_p,
    lax.reduce_max_p,
    lax.reduce_min_p,
    lax.reduce_sum_p,
    lax.reshape_p,
    lax.rev_p,
    lax.round_p,
    lax.slice_p,
    lax.squeeze_p,
    lax.stop_gradient_p,
    lax.transpose_p,
]

_nounit_ops: list[Primitive] = [
    #
    lax.exp2_p,
    lax.exp_p,
    lax.expm1_p,
    lax.log1p_p,
    lax.log_p,
    lax.logistic_p,
    lax.tanh_p,
    #
    lax.acos_p,
    lax.acosh_p,
    lax.asin_p,
    lax.asinh_p,
    lax.atan2_p,
    lax.atan_p,
    lax.atanh_p,
    lax.cos_p,
    lax.cosh_p,
    lax.sin_p,
    lax.sinh_p,
    lax.tan_p,
    ##
    lax.iota_p,
    lax.pow_p,  # maybe special case
    ##
    lax.and_p,
    lax.not_p,
    lax.or_p,
    lax.reduce_and_p,
    lax.reduce_or_p,
    lax.reduce_xor_p,
    lax.xor_p,
    ##
    check_p,
    ##
    lax.rem_p,
    #
    lax.cumprod_p,
]


_mul_ops: list[Primitive] = [
    lax.dot_general_p,
    lax.mul_p,
]

_sametype_ops: list[Primitive] = [
    ad_util.add_any_p,
    lax.add_p,
    lax.clamp_p,
    lax.concatenate_p,
    lax.max_p,
    lax.min_p,
    lax.pad_p,
    lax.sub_p,
]

_sametype_nounit_ret: list[Primitive] = [
    lax.eq_p,
    lax.eq_to_p,
    lax.ge_p,
    lax.gt_p,
    lax.le_p,
    lax.le_to_p,
    lax.lt_p,
    lax.lt_to_p,
    lax.ne_p,
]


@rules_complex(lax.convert_element_type_p)
def _(x: Qt, new_dtype: np.dtype, **kwargs) -> Qt:
    assert isinstance(new_dtype, np.dtype)
    if (
        x._unit not in [dimensionless, anyunit]
        and not np.issubdtype(x.dtype, np.integer)
        and np.issubdtype(new_dtype, np.integer)
    ):
        raise PintaxError_forward(
            ex_type=PintaxNotImplementedError,
            msg="pintax for integers are not yet supported",
        )
    ans = lax.convert_element_type_p.bind(x._val, new_dtype=new_dtype, **kwargs)
    return quantity(ans, x._unit)


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


@rules.many(_mul_ops)
def _(prim: Primitive):
    def func(x: Unit, y: Unit, **kwargs):
        return mul_units(x, y)

    return func


@rules(lax.div_p)
def _(x: Unit, y: Unit):
    return div_units(x, y)


@rules(lax.sign_p)
def _(x: Unit):
    return dimensionless


@rules_complex.many(_sametype_ops)
def _(prim: Primitive):
    assert not prim.multiple_results

    def func(*args: Qt, **kwargs):
        u, args_ = sync_dims(*args)
        ans = prim.bind(*(x._val for x in args_), **kwargs)
        return quantity(ans, u)

    return func


@rules_complex.many(_sametype_nounit_ret)
def _(prim: Primitive):
    assert not prim.multiple_results

    def func(*args: Qt, **kwargs):
        _, args_ = sync_dims(*args)
        ans = prim.bind(*(x._val for x in args_), **kwargs)
        return quantity(ans, dimensionless)

    return func


@rules(lax.integer_pow_p)
def _(x: Unit, y: int):
    if x == anyunit:
        assert y > 0
        return anyunit
    return check_unit(x**y)


@rules(lax.sqrt_p)
def _(x: Unit):
    return pow_unit(x, 1 / 2)


@rules(lax.cbrt_p)
def _(x: Unit):
    return pow_unit(x, 1 / 3)


@rules(lax.square_p)
def _(x: Unit):
    if x == anyunit:
        return anyunit
    return mul_units(x, x)


@rules(lax.split_p)
def _(x: Unit, sizes: Sequence[int], **_):
    return tuple(x for _ in sizes)


@rules(lax.argmin_p)
def _(x: Unit, **_):
    return dimensionless


@rules(lax.argmax_p)
def _(x: Unit, **_):
    return dimensionless


@rules(lax.dynamic_slice_p)
def _(operand: Unit, *starts_and_dyn_sizes: Unit, **_):
    for x in starts_and_dyn_sizes:
        dimensionless_or_err(x)
    return operand


@rules(lax.gather_p)
def _(operand: Unit, indices: Unit, **_):
    dimensionless_or_err(indices)
    return operand


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


@rules(lax.linalg.svd_p)
def _(operand: Unit, compute_uv: bool, **_):
    if compute_uv:
        return (operand, dimensionless, dimensionless)
    else:
        return (operand,)


@rules_complex(lax.scatter_add_p)
def _(operand: Qt, scatter_indices: Qt, updates: Qt, **kwargs):
    dimensionless_or_err(scatter_indices._unit)
    u, (operand, updates) = sync_dims(operand, updates)
    ans = lax.scatter_add_p.bind(
        operand._val, scatter_indices._val, updates._val, **kwargs
    )
    return quantity(ans, u)


@rules_complex(lax.select_n_p)
def _(which: Qt, *cases: Qt):
    assert which._unit in [dimensionless, anyunit]
    u, cases = sync_dims(*cases)
    ans = lax.select_n_p.bind(which._val, *(c._val for c in cases))
    return quantity(ans, u)


@rules(lax.device_put_p)
def _(*args: Unit, **_):
    return args


@rules_complex(lax.scatter_p)
def _(operand: Qt, indices: Qt, updates: Qt, **kwargs) -> Qt:
    dimensionless_or_err(indices._unit)
    u, (operand, updates) = sync_dims(operand, updates)
    ans = lax.scatter_p.bind(operand._val, indices._val, updates._val, **kwargs)
    return quantity(ans, u)


@rules(lax.sort_p)
def _(*args: Unit, **_):
    return args


@rules(lax.linalg.lu_p)
def _(x: Unit, **_):
    # raise PintaxError_forward(ex_type=PintaxTypeError, msg="lu_p is not supported")
    return pow_unit(x, 1 / 2), dimensionless, dimensionless


@rules_complex(lax.reduce_prod_p)
def _(x: Qt, /, *, axes: tuple[int, ...], **kwargs):
    p = math.prod(x.shape[i] for i in axes)
    ans_u = anyunit if x._unit is anyunit else check_unit(x._unit**p)
    ans = lax.reduce_prod_p.bind(x._val, axes=axes, **kwargs)
    return quantity(ans, ans_u)


@rules(lax.linalg.triangular_solve_p)
def _(a: Unit, b: Unit, **_):
    return div_units(b, a)
