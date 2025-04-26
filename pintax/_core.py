from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import partial
from types import TracebackType
from typing import Callable, Concatenate, Final, Literal, final

import equinox as eqx
import jax._src.pretty_printer as pp
import numpy as np
import pint
from jax import Array, lax
from jax._src import core
from jax._src import linear_util as lu
from jax._src import traceback_util
from jax._src.core import Primitive
from jax._src.custom_derivatives import CustomJVPCallPrimitive
from jax._src.dtypes import float0
from jax._src.typing import ArrayLike, DType, Shape
from pint import UnitRegistry
from pint.errors import DimensionalityError, PintTypeError

from ._utils import (
    check_arraylike,
    check_unit,
    dtype_of,
    ensure_jax,
    jit,
    pp_join,
    pp_nested,
    pp_obj,
    pretty_print,
    property_method,
    ruleset,
    shape_of,
    unreachable,
)

traceback_util.register_exclusion(__file__)


@final
class AnyUnit(Enum):
    ANY = 1

    @property
    def dimensionality(self):
        return self


anyunit: Literal[AnyUnit.ANY] = AnyUnit.ANY

Unit = pint.Unit | AnyUnit

global_ureg = UnitRegistry()
dimensionless = global_ureg.dimensionless


def pp_unit(u: Unit) -> pp.Doc:
    return pp.color(
        pp.text(repr(str(u))),
        foreground=pp.Color.BLUE,
        intensity=pp.Intensity.BRIGHT,
    )


def pint_registry() -> UnitRegistry:
    return global_ureg


class PintaxError(Exception):
    pass


class PintaxRuntimeError(PintaxError, RuntimeError):
    pass


class PintaxTypeError(PintaxError, PintTypeError):
    pass


class PintaxDimensionalityError(  # pyright: ignore[reportUnsafeMultipleInheritance]
    PintaxTypeError, DimensionalityError
):
    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, _units1: Unit, _units2: Unit
    ):
        self._units1 = _units1
        self._units2 = _units2

        self._msg = pp_nested(
            "Cannot convert from",
            pp_nested(pp_unit(_units1), f"({_units1.dimensionality})"),
            "to",
            pp_nested(pp_unit(_units2), f"({_units2.dimensionality})"),
        )

    def _with_msg(self, new_msg: pp.Doc) -> PintaxDimensionalityError:
        self._msg = new_msg
        return self

    def _forward(self):
        return PintaxError_forward(ex_type=self._with_msg, msg=self._msg)

    def __str__(self):
        return self._msg.format()


class PintaxNotImplementedError(PintaxError, NotImplementedError):
    pass


class PintaxZeroDivisionError(PintaxError, ZeroDivisionError):
    pass


class PintaxError_forward(Exception):
    def __init__(
        self,
        *,
        ex_type: Callable[[pp.Doc], PintaxError] = PintaxRuntimeError,
        msg: str | pp.Doc,
        throw_from: Exception | None = None,
    ):
        super().__init__()
        self.ex_type = ex_type
        if isinstance(msg, str):
            msg = pp.text(msg)
        self.msg = msg
        self.throw_from = throw_from

    def unwrap(self, msg: str | pp.Doc | None = None):
        if msg is None:
            msg = self.msg
        elif isinstance(msg, str):
            msg = pp.text(msg)
        ans = self.ex_type(msg)
        if ans.__traceback__ is None and self.__traceback__ is not None:
            tb = self.__traceback__
            while tb.tb_next is not None:
                tb = tb.tb_next
            frame = tb.tb_frame
            ans.__traceback__ = TracebackType(
                None, frame, tb_lasti=frame.f_lasti, tb_lineno=frame.f_lineno
            )

        return ans

    @staticmethod
    def from_ex(ex: Exception):
        try:
            msg = pp_join(
                pp_join(
                    "(",
                    pp.color(pp.text(type(ex).__name__), foreground=pp.Color.RED),
                    ")",
                    sep="",
                ),
                *(str(ex).splitlines()),
            )
        except Exception as fmt_ex:
            msg = pp.text(f"<{fmt_ex} while formatting exception that caused this>")

        return PintaxError_forward(msg=msg, throw_from=ex)


@final
class Qt(eqx.Module):
    """
    internal type used to write rules.
    unlike Quantity, unit is always known
    """

    # _val is allowed to be a UnitTracer
    # _val is never allowed to be a Qt
    # _unit must always be a Unit
    _val: ArrayLike
    _unit: Unit = eqx.field(static=True)

    def __init__(self, *, _val: ArrayLike, _unit: Unit):
        super().__init__()
        assert isinstance(_unit, Unit)
        assert not isinstance(_val, Qt)
        self._val = _val
        self._unit = _unit

    @property
    def aval(self) -> core.AbstractValue:
        return core.get_aval(self._val)

    @property
    def dtype(self) -> DType:
        return dtype_of(self._val)

    @property
    def shape(self) -> Shape:
        return shape_of(self._val)

    def _to(self, new_unit: Unit) -> Qt:
        if self._unit == new_unit:
            return self
        if self._unit == anyunit:
            return quantity(self._val, new_unit)
        if new_unit == anyunit:
            raise TypeError()

        return self._to_slow(new_unit)

    @partial(jit, static_argnames=["new_unit"], inline=True)
    def _to_slow(self, new_unit: pint.Unit) -> Qt:
        assert self._unit != anyunit
        dtype = dtype_of(self._val)
        try:
            ans = global_ureg.Quantity(self._val, self._unit).to(new_unit)
        except pint.DimensionalityError:
            raise PintaxDimensionalityError(self._unit, new_unit)._forward() from None
        except pint.PintError as e:
            raise PintaxError_forward.from_ex(e) from None
        assert ans.units == new_unit
        assert isinstance(ans.magnitude, Array)
        if ans.magnitude.dtype == dtype:
            out_mag = ans.magnitude
        else:
            if not np.issubdtype(dtype, np.inexact) and np.issubdtype(
                ans.magnitude.dtype, np.inexact
            ):
                raise PintaxError_forward(
                    ex_type=PintaxTypeError,
                    msg=pp_nested(
                        "not possible to convert an integral value",
                        pretty_print(self._val),
                        "of",
                        pp_unit(self._unit),
                        "to",
                        pp_unit(new_unit),
                    ),
                )
            out_mag = lax.convert_element_type(ans.magnitude, dtype)
        return quantity(out_mag, new_unit)

    @staticmethod
    def _create(v: ArrayLike) -> Qt:

        if isinstance(v, int | float) and (v == 0.0 or math.isinf(v) or math.isnan(v)):
            return quantity(v, anyunit)

        if (
            isinstance(v, np.ndarray)
            and v.shape == ()
            and v.dtype != np.bool
            and (v.dtype == float0 or v == 0.0 or np.isinf(v) or np.isnan(v))
        ):
            return quantity(v, anyunit)

        return quantity(v, dimensionless)

    def _pretty_print(self, prefix: pp.Doc = pp.text("_Quantity_internal")) -> pp.Doc:
        return pp_obj(prefix, pretty_print(self._val), pp_unit(self._unit))

    def __repr__(self):
        return self._pretty_print().format()


def quantity(val: ArrayLike, unit: Unit):
    return Qt(_val=val, _unit=unit)


class UnitTracer(core.Tracer):
    _q: Qt
    _trace: UnitTrace  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, trace: UnitTrace, val: Qt):
        super().__init__(trace)
        assert isinstance(trace, UnitTrace)
        assert isinstance(val, Qt)
        if isinstance(val._val, UnitTracer):
            assert val._val._trace is not trace
        self._q = val

    @property
    def magnitude(self) -> ArrayLike:
        return self._q._val

    @property
    def units(self) -> Unit:
        return self._q._unit

    m = magnitude
    u = units

    @property
    def aval(self):
        return core.get_aval(self._q._val)

    def _contents(self):
        return [
            ("val", self._q._val),
            ("unit", self._q._unit),
        ]

    def full_lower(self):
        return self

    def _pp_custom(self):
        return self._q._pretty_print(pp.text("UnitTracer"))

    def __repr__(self):
        return self._pp_custom().format()

    def _force_dimensionless(self) -> ArrayLike:
        try:
            return self._q._to(dimensionless)._val
        except PintaxError_forward as e:
            raise e.unwrap() from None

    def _ensure_jax(self, x: ArrayLike) -> Array:
        with core.set_current_trace(self._trace._parent):
            return ensure_jax(x)

    def _force_dimensionless_arr(self) -> Array:
        return self._ensure_jax(self._force_dimensionless())

    def to_concrete_value(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> ArrayLike | None:
        if self._q._unit in [dimensionless, anyunit]:
            v = self._q._val
            if isinstance(v, core.Tracer):
                return v.to_concrete_value()
            return v
        return None

    @property_method
    def __array__(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._force_dimensionless_arr().__array__

    @property_method
    def __float__(self):
        return self._force_dimensionless_arr().__float__

    @property_method
    def __bool__(self):
        return self._force_dimensionless_arr().__bool__

    @property_method
    def __int__(self):
        return self._force_dimensionless_arr().__int__

    @property_method
    def item(self):
        return self._force_dimensionless_arr().item

    @property_method
    def tolist(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._force_dimensionless_arr().tolist

    @staticmethod
    def _unpickle(m: ArrayLike, u_: str | AnyUnit) -> Array:
        from ._primitives import prim_mul_unit

        if isinstance(u_, str):
            u = global_ureg.Unit(u_)
        elif u_ is anyunit:
            u = u_
        else:
            unreachable(u_)
        return prim_mul_unit(m, u)

    def __reduce__(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        if self._q._unit is anyunit:
            u_ = anyunit
        else:
            u_ = str(self._q._unit)
        return self._unpickle, (self._q._val, u_)


_no_wrap_prims = {lax.device_put_p}


@dataclass
class _pintax_raise:
    ex: Exception
    from_ex: Exception | None

    def do_raise(self):
        __tracebackhide__ = True
        raise self.ex from self.from_ex

    def __repr__(self):
        return "_pintax_raise"


class UnitTrace(core.Trace[UnitTracer]):

    def __init__(self, parent: core.Trace):
        self._parent = parent
        super().__init__()

    def handle_primitive_arg(self, x: ArrayLike) -> Qt:
        assert not isinstance(x, Qt)
        x = check_arraylike(x)

        if isinstance(x, UnitTracer) and x._trace is self:
            return x._q

        return Qt._create(x)

    def _process_primitive(
        self, primitive: Primitive, tracers: Sequence[ArrayLike], params
    ) -> tuple[ArrayLike, ...] | ArrayLike:

        # print("process_primitive", primitive, tracers, params)

        # if primitive is pjit.pjit_p:
        #     with core.set_current_trace(self):
        #         return tuple(debug_pjit_direct(*tracers, **params))

        with core.set_current_trace(self):
            args = [self.handle_primitive_arg(x) for x in tracers]

        with core.set_current_trace(self._parent):

            # note: this optimization is not valid for
            # make_unit_p, value_and_unit_p
            # and operations invloving jaxpr that can contain those
            if primitive in _no_wrap_prims and all(
                x._unit == dimensionless for x in args
            ):
                return primitive.bind(*(x._val for x in args), **params)

            if primitive in rules_complex:
                out_quants = rules_complex[primitive](self, *args, **params)
            elif primitive in rules:
                out_units = rules[primitive](self, *(x._unit for x in args), **params)

                out_val = primitive.bind(*(x._val for x in args), **params)
                if not primitive.multiple_results:
                    out_val = (out_val,)
                assert len(out_val) == len(out_units)

                out_quants = [quantity(x, u) for u, x in zip(out_units, out_val)]

            else:
                # need more logic to do this optimization; see above comment
                # if all(
                #     x._unit == dimensionless or x._unit == symbolic_zero for x in args
                # ):
                #     return primitive.bind(*(x._val for x in args), **params)

                raise PintaxError_forward(
                    ex_type=PintaxNotImplementedError,
                    msg=f"pintax rule for {repr(primitive)} is not implemented",
                )

            # we cant convert dimensionless back to constant: doing so might cause
            # 0.0 dimentionless to fall into symbolic_zero
            # out_tracers = tuple(
            #     (x._val if x._unit == dimensionless else UnitTracer(self, x))
            #     for x in out_quants
            # )
            out_tracers = tuple(UnitTracer(self, x) for x in out_quants)
            return out_tracers if primitive.multiple_results else out_tracers[0]

    def process_primitive(
        self, primitive: Primitive, tracers: Sequence[ArrayLike], params
    ) -> tuple[ArrayLike, ...] | ArrayLike:
        try:
            return self._process_primitive(primitive, tracers, params)
        except PintaxError:
            raise
        except Exception as ex:

            if not isinstance(ex, PintaxError_forward):
                ex = PintaxError_forward.from_ex(ex)

            parts = [
                pp_nested(pp.text(f"failed to process primitive {primitive}:"), ex.msg)
            ]
            if len(tracers) > 0:
                parts.append(
                    pp_nested(
                        pp.text("with args:"), *[pretty_print(x) for x in tracers]
                    )
                )
            if len(params) > 0:
                parts.append(
                    pp_nested(
                        pp.text("and kwargs:"),
                        *[
                            pp.text(f"{name}=") + pretty_print(val)
                            for name, val in params.items()
                        ],
                    )
                )

            desc = pp.join(pp.brk(), parts)
            # put to seperate function to prevent tools like pytest to display this whole function
            unreachable(_pintax_raise(ex.unwrap(desc), ex.throw_from).do_raise())

    def process_custom_jvp_call(
        self,
        primitive: CustomJVPCallPrimitive,
        fun: lu.WrappedFun,
        jvp: lu.WrappedFun,
        tracers: tuple,
        *,
        symbolic_zeros: bool,
    ):
        # TODO: ?
        del primitive, jvp, symbolic_zeros
        with core.set_current_trace(self):
            return fun.call_wrapped(*tracers)


rules = ruleset[Unit]()
rules_complex = ruleset[Qt]()


@contextlib.contextmanager
def with_unit_trace_():
    with core.take_current_trace() as parent:
        assert parent is not None
        trace = UnitTrace(parent)
        _ctx = core.set_current_trace(trace)
        with _ctx:
            ans = yield trace
            del trace
            _ctx.check_leaks = True  # pyright: ignore[reportAttributeAccessIssue]
            return ans


def with_unit_trace[**P, R](
    f: Callable[Concatenate[UnitTrace, P], R],
) -> Callable[P, R]:

    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        __tracebackhide__ = True
        with with_unit_trace_() as trace:
            ans = f(trace, *args, **kwargs)
            del trace, args, kwargs
            return ans

    return inner


def debug_pjit_direct(*args, jaxpr: core.ClosedJaxpr, **_):
    assert len(jaxpr.consts) == 0
    return core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)


def is_multiplicative(u: Unit) -> bool:
    if u == anyunit:
        return True
    assert u._REGISTRY == global_ureg
    return all(global_ureg._is_multiplicative(name) for name in u._units.keys())


def as_multiplicative(u: Unit) -> Unit:
    if u == anyunit:
        return u
    nonmul = [
        name for name in u._units.keys() if not global_ureg._is_multiplicative(name)
    ]
    if len(nonmul) == 0:
        return u
    assert len(nonmul) == 1
    delta_u = u._units.rename(nonmul[0], "delta_" + nonmul[0])
    return global_ureg.Unit(delta_u)


def assert_multiplicative(
    x: Qt | Unit, ex_type: Callable[[pp.Doc], PintaxError] = PintaxTypeError
):
    if isinstance(x, Qt):
        if not is_multiplicative(x._unit):
            raise PintaxError_forward(
                ex_type=ex_type,
                msg=pp_nested(
                    "quantity with non multiplicative unit:", x._pretty_print()
                ),
            )
    else:
        if not is_multiplicative(x):
            raise PintaxError_forward(
                ex_type=ex_type,
                msg=pp_nested("non multiplicative unit:", pretty_print(x)),
            )


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


def inv_unit(x: Unit) -> Unit:
    if x is anyunit:
        return anyunit
    return check_unit(x ** (-1))


def pow_unit(x: Unit, i: int | float):
    if x == anyunit:
        return anyunit
    return check_unit(x**i)
