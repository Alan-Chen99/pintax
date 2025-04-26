from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    TypeGuard,
    final,
    overload,
)

import equinox as eqx
import jax._src.pretty_printer as pp
import numpy as np
import pint
from jax import Array, lax
from jax import numpy as jnp
from jax import tree_util as jtu
from jax._src import core, traceback_util
from jax._src.numpy.util import promote_args
from jax._src.traceback_util import api_boundary
from jax._src.typing import DType, Shape
from jax.typing import ArrayLike

from ._core import (
    Qt,
)
from ._core import Unit as CUnit_impl
from ._core import (
    UnitTrace,
    UnitTracer,
    anyunit,
    dimensionless,
    div_units,
    global_ureg,
    is_multiplicative,
    mul_units,
    pp_unit,
    with_unit_trace,
)
from ._primitives import prim_convert_unit, prim_mul_unit, prim_value_and_unit
from ._utils import (
    arraylike_to_float,
    cast_unchecked,
    check_arraylike,
    check_unit,
    dtype_of,
    ensure_jax,
    pp_obj,
    pretty_print,
    shape_of,
    tree_map,
    unreachable,
)

traceback_util.register_exclusion(__file__)


def check_QuantityLike(x: Any) -> TypeGuard[QuantityLike]:
    return isinstance(x, QuantityLike)


def ensure_arraylike(x: QuantityLike) -> ArrayLike:
    if isinstance(x, Unit | Quantity):
        return x.asarray()
    return check_arraylike(x)


@overload
def quantity_to_arr(x: QuantityLike) -> Array: ...
@overload
def quantity_to_arr(x: tuple[QuantityLike, ...]) -> tuple[Array, ...]: ...
@overload
def quantity_to_arr(x: list[QuantityLike]) -> list[Array]: ...
@overload
def quantity_to_arr[T](x: T) -> T: ...


def quantity_to_arr(x):
    r"""
    convert :class:`QuantityLike` or a tree of :class:`QuantityLike`
    under :func:`unitify` into :class:`jax.Array`
    """
    return tree_map(_quantity_to_arr, x, is_leaf=check_QuantityLike)


def _quantity_to_arr(x: QuantityLike) -> Array:
    if isinstance(x, Unit | Quantity):
        return x.asarray()
    return prim_mul_unit(x, dimensionless)


class _ConcreteUnit(eqx.Module):
    _u: CUnit_impl = eqx.field(static=True)

    def _pretty_print(self) -> pp.Doc:
        return pp_unit(self._u)

    def __pow__(self, i: int):
        if self._u is anyunit:
            return _ConcreteUnit(anyunit)
        return _ConcreteUnit(check_unit(self._u**i))


class _TracedUnit(eqx.Module):
    _arr: Array

    def _pretty_print(self) -> pp.Doc:
        if isinstance(self._arr, UnitTracer):
            return pp_unit(self._arr._q._unit)
        return pretty_print(self._arr)

    def __pow__(self, i: int):
        return _TracedUnit(self._arr**i)


def _quantity_binop(
    fun: Callable[[Array, Array], Array], reverse=False
) -> Callable[[QuantityLike, QuantityLike], Quantity]:

    # @functools.wraps(fun)
    @api_boundary
    def inner(x1: QuantityLike, x2: QuantityLike, /) -> Quantity:
        if reverse:
            x1, x2 = x2, x1

        if any(_is_traced(x) for x in (x1, x2)):
            x1, x2 = (ensure_arraylike(x) for x in (x1, x2))
            x1, x2 = promote_args(str(fun), x1, x2)
            ans = fun(x1, x2)
            assert isinstance(ans, Array)
            return Quantity._from_arr(ans)

        @with_unit_trace
        def inner2(trace: UnitTrace, x1=x1, x2=x2):
            x1, x2 = (ensure_arraylike(x) for x in (x1, x2))
            x1, x2 = promote_args(str(fun), x1, x2)
            ans = fun(x1, x2)
            return Quantity._from_qt(trace.handle_primitive_arg(ans))

        return inner2()

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
        return Unit._traced(x1.asarray() * x2.asarray())

    return Unit._concrete(mul_units(x1_i._u, x2_i._u))


def _div_unit(x1: Unit, x2: Unit) -> Unit:
    x1_i = x1._impl
    x2_i = x2._impl
    if isinstance(x1_i, _TracedUnit) or isinstance(x2_i, _TracedUnit):
        return Unit._traced(x1.asarray() / x2.asarray())

    return Unit._concrete(div_units(x1_i._u, x2_i._u))


def _parse_QuantityLike(x: QuantityLike) -> tuple[ArrayLike | None, Unit | None]:
    if isinstance(x, Unit):
        return None, x
    elif isinstance(x, Quantity):
        return x.m, x.u
    else:
        x = check_arraylike(x)
        if isinstance(x, Array):
            m, u = prim_value_and_unit(x)
            return m, Unit._traced(u)
        return x, None


def _unparse_QuantityLike(x_m: ArrayLike | None, x_u: Unit | None) -> QuantityLike:
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


def _quantity_comp_op(
    fun: Callable[[Array, Array], Array], reverse=False
) -> Callable[[QuantityLike, QuantityLike], Array]:

    @functools.wraps(fun)
    @api_boundary
    def inner(x1: QuantityLike, x2: QuantityLike, /) -> Array:
        if reverse:
            x1, x2 = x2, x1

        if any(_is_traced(x) for x in (x1, x2)):
            x1, x2 = (ensure_arraylike(x) for x in (x1, x2))
            x1, x2 = promote_args(str(fun), x1, x2)
            ans = fun(x1, x2)
            assert isinstance(ans, Array)
            return ans

        @with_unit_trace
        def inner2(trace: UnitTrace, x1=x1, x2=x2):
            x1, x2 = (ensure_arraylike(x) for x in (x1, x2))
            x1, x2 = promote_args(str(fun), x1, x2)
            ans = fun(x1, x2)
            ans_ = trace.handle_primitive_arg(ans)
            assert ans_._unit in [dimensionless, anyunit]
            return ans_._val

        return jnp.array(inner2())

    return inner


def arraymethod_linear[**P](method: Callable[[Array], Callable[P, Array]]):
    @api_boundary
    def inner(self: Quantity, *args: P.args, **kwargs: P.kwargs) -> Quantity:
        ans = method(ensure_jax(self.m))(*args, **kwargs)
        return Quantity._create(ans, self.u)

    setattr(inner, "__signature__", inspect.signature(method(cast_unchecked()(Array))))
    return inner


@final
@dataclass
class Unit:
    """
    a unit. can be used inside and outside of :func:`unitify`.
    """

    _impl: _ConcreteUnit | _TracedUnit

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
            return core.get_aval(1)
        elif isinstance(x, _TracedUnit):
            return x._arr.aval
        else:
            unreachable(x)

    @property
    def dtype(self) -> DType:
        return dtype_of(self.aval)

    @property
    def shape(self) -> Shape:
        return shape_of(self.aval)

    @api_boundary
    def asarray(self) -> Array:
        x = self._impl
        if isinstance(x, _ConcreteUnit):
            return prim_mul_unit(1 if is_multiplicative(x._u) else jnp.nan, x._u)
        elif isinstance(x, _TracedUnit):
            return x._arr
        else:
            unreachable(x)

    @property
    def a(self) -> Array:
        """
        Get an Array corresponding to 1 of this Unit.
        can only be used under :func:`unitify`.
        """
        return self.asarray()

    def __jax_array__(self) -> Array:
        """
        ``Unit`` can be implicitly converted to a :class:`jax.Array`

        .. code:: python

            @unitify
            def main():
                q = quantity(5.0 * areg.m)

                print(jnp.array([q.u, 2 * q.u]))

                # in a multiplication the left ones __mul__ is used
                # if the left is jax.Array,
                # the unit is converted to jax.Array via __jax_array__
                reveal_type(q.u * q.u)  # Unit
                reveal_type(jnp.sin(q.m) * q.u)  # Array
                reveal_type(q.u * jnp.sin(q.m))  # Quantity
                reveal_type(2.0 * q.u)  # Quantity

                # TypeError at runtime: not currently possible for Quantity to implement __jax_array__
                _ = jnp.sin(q.m) * q
        """
        return self.asarray()

    def _pretty_print(self) -> pp.Doc:
        return self._impl._pretty_print()

    def __repr__(self):
        return pp_obj("Unit", self._pretty_print()).format()

    __add__ = _quantity_binop(lax.add)
    __radd__ = _quantity_binop(lax.add, reverse=True)

    __sub__ = _quantity_binop(lax.sub)
    __rsub__ = _quantity_binop(lax.sub, reverse=True)

    __mul__ = _unit_mul_div(_mul_quantity)
    __rmul__ = _unit_mul_div(_mul_quantity, reverse=True)

    __truediv__ = _unit_mul_div(_div_quantity)
    __rtruediv__ = _unit_mul_div(_div_quantity, reverse=True)

    def __pow__(self, i: int, /) -> Unit:
        return Unit(self._impl**i)

    def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: QuantityLike, /
    ) -> Array:
        return _quantity_comp_op(lax.eq)(self, other)


class Quantity(eqx.Module):
    """
    A Quantity type used to pass data between functions under :func:`unitify`.

    It can also be used insdie :func:`unitify` to obtain the value and unit of an array.

    Example:

    .. code:: python

        from functools import partial
        from jax import Array
        from pintax import areg, unitify, ureg

        @partial(unitify, static_typed=False)
        def f(x: Array):
            assert isinstance(x, Array)
            print("x", x)
            ans = x * areg.m
            print("ans", ans)
            return ans

        def main():
            v1 = 1.0 * ureg.m
            # constants with units only work for python scalars and numpy arrays
            # jax arrays does not work
            # _ = jnp.array(1) * ureg.m  # TypeError
            print("v1", v1)
            print()
            v2 = f(v1 * 2)
            print("v2", v2)
            print()
            v3 = f(v2 * 2)
            print("v3", v3)

    prints

    .. code:: text

        v1 Quantity(1.0, 'meter')

        x UnitTracer(2.0, 'meter')
        ans UnitTracer(Array(2., dtype=float32, weak_type=True), 'meter ** 2')
        v2 Quantity(Array(2., dtype=float32, weak_type=True), 'meter ** 2')

        x UnitTracer(Array(4., dtype=float32, weak_type=True), 'meter ** 2')
        ans UnitTracer(Array(4., dtype=float32, weak_type=True), 'meter ** 3')
        v3 Quantity(Array(4., dtype=float32, weak_type=True), 'meter ** 3')

    """

    m: ArrayLike  #: Magnitude
    _unit_impl: _ConcreteUnit | _TracedUnit
    _disable_jvp_marker: ArrayLike = 1

    @property
    def m_arr(self) -> Array:
        return ensure_jax(self.m)

    @property
    def u(self) -> Unit:
        """Unit"""
        return Unit(self._unit_impl)

    __array_priority__ = pint.Quantity.__array_priority__

    def __init__(self, *, _m: ArrayLike, _u: Unit):
        super().__init__()
        self.m = _m
        self._unit_impl = _u._impl

    @staticmethod
    def _create(m: ArrayLike, u: Unit) -> Quantity:
        return Quantity(_m=m, _u=u)

    @staticmethod
    def _from_arr(x: Array) -> Quantity:
        m, u = prim_value_and_unit(x)
        return Quantity(_m=m, _u=Unit._traced(u))

    @property
    def aval(self) -> core.AbstractValue:
        return core.get_aval(self.m)

    @property
    def dtype(self) -> DType:
        return dtype_of(self.m)

    @property
    def shape(self) -> Shape:
        return shape_of(self.aval)

    @property
    def size(self) -> int:
        return jnp.size(self.m)

    @property
    def ndim(self) -> int:
        return jnp.ndim(self.m)

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
    def asarray(self) -> Array:
        if isinstance(self.u._impl, _ConcreteUnit):
            return prim_mul_unit(self.m, self.u._impl._u)

        u_arr = lax.convert_element_type(self.u.asarray(), new_dtype=dtype_of(self.m))
        ans = self.m * u_arr
        assert isinstance(ans, Array)
        return ans

    @property
    def a(self) -> Array:
        """
        converts Quantity into a Array under :func:`unitify`
        """
        return self.asarray()

    @property
    def as_arr_t(self):
        if TYPE_CHECKING:
            # fake assert, will not pass at runtime
            assert isinstance(self, Array)
            return self
        else:
            return self

    # lax does not jit and gives better unit errors
    __add__ = _quantity_binop(lax.add)
    __radd__ = _quantity_binop(lax.add, reverse=True)

    __sub__ = _quantity_binop(lax.sub)
    __rsub__ = _quantity_binop(lax.sub, reverse=True)

    __mul__ = _unit_mul_div(_mul_quantity)
    __rmul__ = _unit_mul_div(_mul_quantity, reverse=True)

    __truediv__ = _unit_mul_div(_div_quantity)
    __rtruediv__ = _unit_mul_div(_div_quantity, reverse=True)

    __lt__ = _quantity_comp_op(lax.lt)
    __le__ = _quantity_comp_op(lax.le)
    __gt__ = _quantity_comp_op(lax.gt)
    __ge__ = _quantity_comp_op(lax.ge)

    def __pow__(self, i: int, /):
        return Quantity._create(self.m**i, self.u**i)

    def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: QuantityLike, /
    ) -> Array:
        return _quantity_comp_op(lax.eq)(self, other)

    def to(self, new_unit: Unit, /) -> Quantity:
        assert isinstance(new_unit, Unit)

        if _is_traced(self) or _is_traced(new_unit):
            ans = prim_convert_unit(self.asarray(), new_unit.asarray())
            return Quantity._from_arr(ans)

        @with_unit_trace
        def inner2(trace: UnitTrace):
            x1 = ensure_arraylike(self)
            x2 = ensure_arraylike(new_unit)
            ans = prim_convert_unit(x1, x2)
            return Quantity._from_qt(trace.handle_primitive_arg(ans))

        return inner2()

    def _m_dimensionless(self) -> ArrayLike:
        return self.to(Unit._concrete(dimensionless)).m

    @property
    @api_boundary
    def __array__(self):
        ans = self._m_dimensionless()
        return ensure_jax(ans).__array__

    @api_boundary
    def __float__(self) -> float:
        return float(arraylike_to_float(self._m_dimensionless()))

    @api_boundary
    def __bool__(self) -> bool:
        return bool(self._m_dimensionless())

    @api_boundary
    def __getitem__(self, key) -> Quantity:
        return arraymethod_linear(lambda x: x.__getitem__)(self, key)

    flatten = arraymethod_linear(lambda x: x.flatten)
    ravel = arraymethod_linear(lambda x: x.ravel)
    reshape = arraymethod_linear(lambda x: x.reshape)
    squeeze = arraymethod_linear(lambda x: x.squeeze)
    transpose = arraymethod_linear(lambda x: x.transpose)


QuantityLike = Unit | Quantity | ArrayLike


def _is_traced(x: QuantityLike) -> bool:
    return (
        isinstance(x, Array)
        or (isinstance(x, Unit) and isinstance(x._impl, _TracedUnit))
        or (isinstance(x, Quantity) and isinstance(x.u._impl, _TracedUnit))
    )


@overload
def unitify[**P](
    fun: Callable[P, Array],
    /,
    *,
    unwrap_outs: Literal[False] = False,
    force_dimensionless_outs: Literal[False] = False,
    static_typed: Literal[True] = True,
    wrap_inputs=True,
) -> Callable[P, Quantity]: ...


@overload
def unitify[**P](
    fun: Callable[P, Array],
    /,
    *,
    unwrap_outs: Literal[True],
    force_dimensionless_outs: Literal[False] = False,
    static_typed: Literal[True] = True,
    wrap_inputs=True,
) -> Callable[P, Array | Quantity]: ...


@overload
def unitify[**P](
    fun: Callable[P, Array],
    /,
    *,
    unwrap_outs: Literal[False] = False,
    force_dimensionless_outs: Literal[True],
    static_typed: Literal[True] = True,
    wrap_inputs=True,
) -> Callable[P, Array]: ...


@overload
def unitify[**P, R](
    fun: Callable[P, R],
    /,
    *,
    unwrap_outs=False,
    force_dimensionless_outs=False,
    static_typed: Literal[True] = True,
    wrap_inputs=True,
) -> Callable[P, R]: ...


@overload
def unitify[R](
    fun: Callable[..., R],
    /,
    *,
    unwrap_outs=False,
    force_dimensionless_outs=False,
    static_typed: Literal[False],
    wrap_inputs=True,
) -> Callable[..., R | Any]: ...


def unitify(
    fun: Callable,
    /,
    *,
    unwrap_outs=False,
    force_dimensionless_outs=False,
    static_typed=True,
    wrap_inputs=True,
) -> Callable:
    """
    To use units in a function, it must be wrapped with :func:`unitify`.

    Args:
        fun: function to unitify.
            It may take arbitrary pytree arguments and return a pytree.

        unwrap_outs:
            if True, output as :class:`jax.Array` for
            output buffers that are dimensionless

        force_dimensionless_outs:
            if True, if outputs are not dimensionless, an error is thrown.
            otherwise, output buffers are returned as :class:`jax.Array`

        wrap_inputs:
            if True, convert all instances of Quantity in the args to Array.
            otherwise, inputs are passed as is.

        static_typed:
            if True, overload signature will be enforced by a static typechecker.
            Note that the actual behavior does not exactly match the behavior
            specified by the overloads.


    inputs of type :class:`pintax.Quantity` are seems as `Array` inside ``fun``

    first 3 overloads extend to pytrees;
    last overload infer a generic type that might not be correct.
    """
    del static_typed
    if force_dimensionless_outs and unwrap_outs:
        raise TypeError()

    @functools.wraps(fun)
    @api_boundary
    @with_unit_trace
    def inner(trace: UnitTrace, *args, **kwargs):
        if wrap_inputs:
            args, kwargs = quantity_to_arr((args, kwargs))
        ans = fun(*args, **kwargs)

        def handle_output(x_: QuantityLike):
            x = trace.handle_primitive_arg(ensure_arraylike(x_))

            if unwrap_outs:
                if x._unit in [dimensionless, anyunit]:
                    return x._val
                else:
                    return Quantity._from_qt(x)

            elif force_dimensionless_outs:
                assert x._unit in [dimensionless, anyunit]
                return x._val

            else:
                return Quantity._from_qt(x)

        return tree_map(handle_output, ans, is_leaf=check_QuantityLike)

    try:
        setattr(inner, "__signature__", inspect.signature(fun))
    except:
        pass
    return inner


class _ureg_t:
    @api_boundary
    def __getattr__(self, name: str) -> Unit:
        return self(name)

    @api_boundary
    def __call__(self, name: str) -> Unit:
        ans = check_unit(global_ureg(name).units)
        return Unit._concrete(ans)

    def __repr__(self):
        return f"<{self.__module__}.ureg>"


ureg = _ureg_t()


class _areg_t:
    @staticmethod
    @api_boundary
    def __getattr__(name: str, /) -> Array:
        """
        get the unit from a name

        >>> from pintax import *
        >>> unitify(lambda: areg.m)()
        Quantity(1, 'meter')
        """
        return _areg_t.__call__(name)

    @property
    def dimensionless(self) -> Array:
        return self("dimensionless")

    @staticmethod
    @api_boundary
    def __call__(name: str, /) -> Array:
        """
        get the unit from a name, using a string.

        can only be used under :func:`pintax.unitify`.

        >>> from pintax import *
        >>> unitify(lambda: areg("m"))()
        Quantity(1, 'meter')
        """
        ans = check_unit(global_ureg(name).units)
        return prim_mul_unit(1 if is_multiplicative(ans) else jnp.nan, ans)

    def __repr__(self):
        return f"<{self.__module__}.areg>"


areg = _areg_t()
