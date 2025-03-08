from __future__ import annotations

import functools
from typing import Any, Callable, Sequence, TypeGuard, overload

import equinox as eqx
import jax._src.pretty_printer as pp
import numpy as np
import pint
from jax import Array, lax
from jax._src import core, traceback_util
from jax._src.numpy.util import promote_args
from jax._src.traceback_util import api_boundary
from jax._src.typing import ArrayLike, DType, Shape, StaticScalar

from ._core import (
    Qt,
)
from ._core import Unit as CUnit_impl
from ._core import (
    UnitTracer,
    _global_ureg,
    dimensionless,
    pp_unit,
    with_unit_trace,
)
from ._primitives import convert_unit, make_unit, value_and_unit
from ._utils import (
    arraylike_to_float,
    cast_unchecked,
    check_arraylike,
    check_unit,
    dtype_of,
    flattenctx,
    pp_obj,
    pretty_print,
    unreachable,
    with_flatten,
)

traceback_util.register_exclusion(__file__)


class _ConcreteUnit(eqx.Module):
    _u: CUnit_impl = eqx.field(static=True)

    def _pretty_print(self) -> pp.Doc:
        return pp_unit(self._u)


class _TracedUnit(eqx.Module):
    _arr: Array

    def _pretty_print(self) -> pp.Doc:
        if isinstance(self._arr, UnitTracer):
            return pp_unit(self._arr._q._unit)
        return pretty_print(self._arr)


def _quantity_binop(
    fun: Callable[[Array, Array], Array], reverse=False, promote=True
) -> Callable[[QuantityLike, QuantityLike], Quantity]:

    @functools.wraps(fun)
    @api_boundary
    def inner(x1: QuantityLike, x2: QuantityLike, /) -> Quantity:
        if reverse:
            x1, x2 = x2, x1

        if any(_is_traced(x) for x in (x1, x2)):
            x1, x2 = (_ensure_array(x) for x in (x1, x2))
            x1, x2 = promote_args(str(fun), x1, x2)
            ans = fun(x1, x2)
            assert isinstance(ans, Array)
            return Quantity._from_arr(ans)

        with with_unit_trace() as trace:
            x1, x2 = (_ensure_array(x) for x in (x1, x2))
            x1, x2 = promote_args(str(fun), x1, x2)
            ans = fun(x1, x2)
            return Quantity._from_qt(trace.ensure_quantity(ans))

    return inner


def _unit_mul_div(
    fun: Callable[[QuantityLike, QuantityLike], QuantityLike],
    reverse=False,
):
    @overload
    def inner(x1: Unit, x2: Unit, /) -> Unit: ...

    @overload
    def inner(x1: Quantity | ArrayLike, x2: QuantityLike, /) -> Quantity: ...
    @overload
    def inner(x1: QuantityLike, x2: Quantity | ArrayLike, /) -> Quantity: ...

    @functools.wraps(fun)
    @api_boundary
    def inner(x1: QuantityLike, x2: QuantityLike, /) -> QuantityLike:
        if reverse:
            x1, x2 = x2, x1
        return fun(x1, x2)

    return inner


def _mul_unit(x1: Unit, x2: Unit) -> Unit:
    x1_i = x1._impl
    x2_i = x2._impl
    if isinstance(x1_i, _TracedUnit) or isinstance(x2_i, _TracedUnit):
        return Unit._traced(x1.as_array() * x2.as_array())

    from ._rules import mul_units

    return Unit._concrete(mul_units(x1_i._u, x2_i._u))


def _div_unit(x1: Unit, x2: Unit) -> Unit:
    x1_i = x1._impl
    x2_i = x2._impl
    if isinstance(x1_i, _TracedUnit) or isinstance(x2_i, _TracedUnit):
        return Unit._traced(x1.as_array() / x2.as_array())

    from ._rules import div_units

    return Unit._concrete(div_units(x1_i._u, x2_i._u))


def _parse_QuantityLike(x: QuantityLike) -> tuple[ArrayLike | None, Unit | None]:
    if isinstance(x, Unit):
        return None, x
    elif isinstance(x, Quantity):
        return x.m, x.u
    else:
        check_arraylike(x)
        if isinstance(x, Array):
            m, u = value_and_unit(x)
            return m, Unit._traced(u)
        return x, None


def _unparse_QuantityLike(x_m: ArrayLike | None, x_u: Unit | None) -> QuantityLike:
    if x_m is not None and np.issubdtype(dtype_of(x_m), np.integer):
        x_m = 1.0 * x_m

    if x_m is None and x_u is None:
        raise TypeError()
    elif x_m is None:
        assert x_u is not None
        return x_u
    elif x_u is None:
        return x_m
    else:
        return Quantity._create(x_m, x_u)


def _mul_quantity(x1: QuantityLike, x2: QuantityLike, /) -> QuantityLike:

    x1_m, x1_u = _parse_QuantityLike(x1)
    x2_m, x2_u = _parse_QuantityLike(x2)

    if x1_u is None and x2_u is None:
        ans_u = None
    elif x1_u is None:
        ans_u = x2_u
    elif x2_u is None:
        ans_u = x1_u
    else:
        ans_u = _mul_unit(x1_u, x2_u)

    if x1_m is None and x2_m is None:
        ans_m = None
    elif x1_m is None:
        ans_m = x2_m
    elif x2_m is None:
        ans_m = x1_m
    else:
        ans_m = check_arraylike(x1_m * x2_m)

    return _unparse_QuantityLike(ans_m, ans_u)


def _div_quantity(x1: QuantityLike, x2: QuantityLike, /) -> QuantityLike:

    x1_m, x1_u = _parse_QuantityLike(x1)
    x2_m, x2_u = _parse_QuantityLike(x2)

    if x1_u is None and x2_u is None:
        ans_u = None
    elif x1_u is None:
        assert x2_u is not None
        ans_u = _div_unit(Unit._concrete(dimensionless), x2_u)
    elif x2_u is None:
        ans_u = x1_u
    else:
        ans_u = _div_unit(x1_u, x2_u)

    if x1_m is None and x2_m is None:
        ans_m = None
    elif x1_m is None:
        assert x2_m is not None
        ans_m = lax.div(np.array(1.0, dtype=dtype_of(x2_m)), x2_m)
    elif x2_m is None:
        ans_m = x1_m
    else:
        ans_m = check_arraylike(x1_m / x2_m)

    return _unparse_QuantityLike(ans_m, ans_u)


class Unit(eqx.Module):
    _impl: _ConcreteUnit | _TracedUnit
    _disable_jvp_marker: ArrayLike = 1

    __array_priority__ = pint.Unit.__array_priority__

    @staticmethod
    def _concrete(x: CUnit_impl):
        return Unit(_ConcreteUnit(x))

    @staticmethod
    def _traced(x: Array):
        return Unit(_TracedUnit(x))

    @property
    def aval(self) -> core.AbstractValue:
        x = self._impl
        if isinstance(x, _ConcreteUnit):
            return core.get_aval(1.0)
        elif isinstance(x, _TracedUnit):
            return x._arr.aval
        else:
            unreachable(x)

    @property
    def dtype(self) -> DType:
        return self.aval.dtype  # type: ignore

    @property
    def shape(self) -> Shape:
        return self.aval.shape  # type: ignore

    @api_boundary
    def as_array(self) -> Array:
        x = self._impl
        if isinstance(x, _ConcreteUnit):
            return make_unit(1, x._u)
        elif isinstance(x, _TracedUnit):
            return x._arr
        else:
            unreachable(x)

    __jax_array__ = as_array

    def _pretty_print(self) -> pp.Doc:
        return self._impl._pretty_print()

    def __repr__(self):
        return pp_obj("Unit", self._pretty_print()).format()

    __mul__ = _unit_mul_div(_mul_quantity)
    __rmul__ = _unit_mul_div(_mul_quantity, reverse=True)

    __truediv__ = _unit_mul_div(_div_quantity)
    __rtruediv__ = _unit_mul_div(_div_quantity, reverse=True)


class Quantity(eqx.Module):
    m: ArrayLike
    u: Unit

    __array_priority__ = pint.Quantity.__array_priority__

    def __init__(self, *, _m: ArrayLike, _u: Unit):
        self.m = _m
        self.u = _u

    @staticmethod
    def _create(m: ArrayLike, u: Unit) -> Quantity:
        return Quantity(_m=m, _u=u)

    @staticmethod
    def _from_arr(x: Array) -> Quantity:
        m, u = value_and_unit(x)
        return Quantity(_m=m, _u=Unit._traced(u))

    @property
    def aval(self) -> core.AbstractValue:
        return core.get_aval(self.m)

    @property
    def dtype(self) -> DType:
        return dtype_of(self.m)

    @property
    def shape(self) -> Shape:
        return self.aval.shape  # type: ignore

    @staticmethod
    def _from_qt(qt: Qt) -> Quantity:
        return Quantity._create(qt._val, Unit._concrete(qt._unit))

    def __tree_pp__(self, **kwargs) -> pp.Doc:
        if isinstance(self.m, UnitTracer) and self.m._q._unit == dimensionless:
            val = pretty_print(self.m._q._val)
        else:
            val = pretty_print(self.m)
        unit = self.u._pretty_print()
        return pp_obj("Quantity", val, unit)

    def __repr__(self):
        return self.__tree_pp__().format()

    def __format__(self, format_spec: str, /) -> str:
        if isinstance(self.m, UnitTracer) and self.m._q._unit == dimensionless:
            val = self.m._q._val
        else:
            val = self.m
        val_fmt = val.__format__(format_spec)
        unit = self.u._pretty_print()
        return pp_obj("Quantity", val_fmt, unit).format()

    @api_boundary
    def as_array(self) -> Array:
        if isinstance(self.u._impl, _ConcreteUnit):
            return make_unit(self.m, self.u._impl._u)

        u_arr = lax.convert_element_type(self.u.as_array(), new_dtype=dtype_of(self.m))
        ans = self.m * u_arr
        assert isinstance(ans, Array)
        return ans

    __jax_array__ = as_array

    # lax does not jit and gives better unit errors
    __add__ = _quantity_binop(lax.add)
    __radd__ = _quantity_binop(lax.add, reverse=True)

    __sub__ = _quantity_binop(lax.sub)
    __rsub__ = _quantity_binop(lax.sub, reverse=True)

    __mul__ = _unit_mul_div(_mul_quantity)
    __rmul__ = _unit_mul_div(_mul_quantity, reverse=True)

    __truediv__ = _unit_mul_div(_div_quantity)
    __rtruediv__ = _unit_mul_div(_div_quantity, reverse=True)

    def to(self, new_unit: Unit) -> Quantity:
        if _is_traced(self) or _is_traced(new_unit):
            ans = convert_unit(self.as_array(), new_unit.as_array())
            return Quantity._from_arr(ans)

        with with_unit_trace() as trace:
            x1 = _ensure_array(self)
            x2 = _ensure_array(new_unit)
            ans = convert_unit(x1, x2)
            return Quantity._from_qt(trace.ensure_quantity(ans))

    def _m_dimensionless(self) -> ArrayLike:
        return self.to(Unit._concrete(dimensionless)).m

    @api_boundary
    def __array__(self) -> np.ndarray:
        return np.array(self._m_dimensionless())

    @api_boundary
    def __float__(self) -> float:
        return float(arraylike_to_float(self._m_dimensionless()))


QuantityLike = Unit | Quantity | ArrayLike
ArrayLikeNonArr = np.ndarray | StaticScalar
QuantityLikeNonArr = Unit | Quantity | ArrayLikeNonArr


def _is_traced(x: QuantityLike) -> bool:
    return (
        isinstance(x, Array)
        or (isinstance(x, Unit) and isinstance(x._impl, _TracedUnit))
        or (isinstance(x, Quantity) and isinstance(x.u._impl, _TracedUnit))
    )


def _ensure_array(x: QuantityLike) -> ArrayLike:
    if isinstance(x, Unit | Quantity):
        return x.as_array()
    check_arraylike(x)
    return x


def _check_QuantityLike(x: Any) -> TypeGuard[QuantityLike]:
    return isinstance(x, QuantityLike)


def unitify(
    fun: Callable, *, unwrap_outs=True, force_dimensionless_outs=False
) -> Callable:
    # """
    # first 3 overloads extend to pytrees
    # last overload infer a generic type that might not be correct
    # """
    if force_dimensionless_outs and not unwrap_outs:
        raise TypeError()

    def _unitify_flat(
        ctx: flattenctx[QuantityLike], bufs: Sequence[QuantityLike]
    ) -> Sequence[QuantityLike]:
        with with_unit_trace() as trace:
            for x in bufs:
                if not isinstance(x, Array):
                    assert not _is_traced(x)
            bufs_q = [_ensure_array(x) for x in bufs]
            out_bufs = ctx.call(bufs_q)
            out_bufs_q = [trace.ensure_quantity(_ensure_array(x)) for x in out_bufs]
            if unwrap_outs:
                return [
                    x._val if x._unit == dimensionless else Quantity._from_qt(x)
                    for x in out_bufs_q
                ]
            elif force_dimensionless_outs:
                for x in out_bufs_q:
                    assert x._unit == dimensionless
                return [x._val for x in out_bufs_q]
            else:
                return [Quantity._from_qt(x) for x in out_bufs_q]

    return with_flatten(fun, _unitify_flat, _check_QuantityLike)


class _registry_wrapper:
    @api_boundary
    def __getattr__(self, name: str) -> Unit:
        return self(name)

    @api_boundary
    def __call__(self, name: str) -> Unit:
        ans = check_unit(_global_ureg(name).units)
        return Unit._concrete(ans)


ureg = _registry_wrapper()


class _arr_registry_wrapper:
    @api_boundary
    def __getattr__(self, name: str) -> Array:
        return self(name)

    @api_boundary
    def __call__(self, name: str) -> Array:
        ans = check_unit(_global_ureg(name).units)
        return make_unit(1, ans)


areg = _arr_registry_wrapper()
