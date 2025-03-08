from __future__ import annotations

import contextlib
import math
from enum import Enum
from functools import partial
from typing import Literal, Sequence, final

import equinox as eqx
import jax._src.pretty_printer as pp
import numpy as np
import pint
from jax import Array, lax
from jax._src import core, traceback_util
from jax._src.core import Primitive
from jax._src.typing import ArrayLike, DType, Shape
from pint import UnitRegistry
from pint.errors import PintTypeError

from ._utils import (
    cast_unchecked,
    check_arraylike,
    dtype_of,
    jit,
    pp_join,
    pp_nested,
    pp_obj,
    pretty_print,
    ruleset,
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

_global_ureg = UnitRegistry()
dimensionless = _global_ureg.dimensionless


def pp_unit(u: Unit) -> pp.Doc:
    return pp.color(
        pp.text(repr(str(u))),
        foreground=pp.Color.BLUE,
        intensity=pp.Intensity.BRIGHT,
    )


def pint_registry() -> UnitRegistry:
    return _global_ureg


class PintaxError(Exception):
    pass


class PintaxRuntimeError(PintaxError, RuntimeError):
    pass


class PintaxTypeError(PintaxError, PintTypeError):
    pass


class PintaxNotImplementedError(PintaxError, NotImplementedError):
    pass


class PintaxZeroDivisionError(PintaxError, ZeroDivisionError):
    pass


class PintaxError_forward(Exception):
    ex_type: type[PintaxError]
    msg: pp.Doc | None

    def __init__(
        self,
        *,
        ex_type: type[PintaxError] = PintaxRuntimeError,
        msg: str | pp.Doc | None = None,
    ):
        self.ex_type = ex_type
        if isinstance(msg, str):
            msg = pp.text(msg)
        self.msg = msg


def _unitify_check_ret(x: Qt, trace: UnitTrace) -> Qt:
    if isinstance(x._val, UnitTracer) and x._val._trace is trace:
        raise TypeError()
    if isinstance(x._val, Qt):
        raise TypeError()
    return x


# def _quantity_binop(
#     fun: Callable[[Array, Array], Array], reverse=False
# ) -> Callable[[QuantityLike, QuantityLike], Qt]:

#     @functools.wraps(fun)
#     @api_boundary
#     def inner(x1: QuantityLike, x2: QuantityLike, /) -> Qt:
#         if reverse:
#             x1, x2 = x2, x1
#         for x in [x1, x2]:
#             if isinstance(x, Array):
#                 raise TypeError(f"expected non-jax constant, got\n{x}")
#         # if any(isinstance(x, Array) for x in (x1, x2)):
#         #     raise TypeError()
#         #     # x1, x2 = (Quantity._create(x).as_array() for x in (x1, x2))
#         #     # x1, x2 = promote_args(str(fun), x1, x2)
#         #     # return fun(x1, x2)
#         with with_unit_trace() as trace:
#             x1, x2 = (UnitTracer(trace, trace.ensure_quantity(x)) for x in (x1, x2))
#             x1, x2 = promote_args(str(fun), x1, x2)
#             ans = fun(x1, x2)
#             return trace.ensure_quantity(ans)

#     return inner


@final
class Qt(eqx.Module):
    # _val is allowed to be a UnitTracer
    # _val is never allowed to be a Quantity
    # _unit must always be a Unit
    _val: ArrayLike
    _unit: Unit = eqx.field(static=True)

    def __init__(self, *, _val: ArrayLike, _unit: Unit):
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
        return self.aval.shape  # type: ignore

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
        assert not np.issubdtype(dtype, np.integer)
        try:
            ans = _global_ureg.Quantity(self._val, self._unit).to(new_unit)
        except pint.DimensionalityError as e:
            raise PintaxError_forward(
                ex_type=PintaxTypeError,
                msg=str(e),
            ) from None
        assert ans.units == new_unit
        assert isinstance(ans.magnitude, Array)
        if ans.magnitude.dtype == dtype:
            out_mag = ans.magnitude
        else:
            out_mag = lax.convert_element_type(ans.magnitude, dtype)
        return quantity(out_mag, new_unit)

    @staticmethod
    def _create(v: _QuantityLike) -> Qt:
        if isinstance(v, Qt):
            return v
        # if isinstance(v, pint.Quantity):
        #     unit = v.units
        #     assert isinstance(unit, Unit)
        #     check_arraylike(v.magnitude)
        #     return quantity(v.magnitude, unit)
        # if isinstance(v, Unit):
        #     return quantity(1.0, v)

        if isinstance(v, int | float) and (v == 0.0 or math.isinf(v) or math.isnan(v)):
            return quantity(v, anyunit)

        if (
            isinstance(v, np.ndarray)
            and v.shape == ()
            and (v == 0.0 or np.isinf(v) or np.isnan(v))
        ):
            return quantity(v, anyunit)

        check_arraylike(v)
        return quantity(v, dimensionless)

    def _pretty_print(self, prefix: pp.Doc = pp.text("_Quantity_internal")) -> pp.Doc:
        val = pretty_print(self._val)
        return pp_obj(
            prefix,
            pretty_print(self._val),
            pp_unit(self._unit),
        )

    def __repr__(self):
        return self._pretty_print().format()


def quantity(val: ArrayLike, unit: Unit):
    return Qt(_val=val, _unit=unit)


_QuantityLike = ArrayLike | Qt


class UnitTracer(core.Tracer):
    _q: Qt

    def __init__(self, trace: core.Trace, val: Qt):
        assert isinstance(val, Qt)
        if isinstance(val._val, UnitTracer):
            assert val._val._trace is not trace
        self._trace = trace
        self._q = val

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

    def __repr__(self):
        return self._q._pretty_print(pp.text("UnitTracer")).format()

    def __array__(self):
        ans = self._q._to(dimensionless)._val
        return np.array(ans)

    def __float__(self):
        ans = self._q._to(dimensionless)._val
        return float(cast_unchecked()(ans))


class UnitTrace(core.Trace[UnitTracer]):

    def __init__(self, parent: core.Trace):
        self._parent = parent

    def invalidate(self):
        print("invalidate:", self, id(self))
        super().invalidate()

    def ensure_quantity(self, x: _QuantityLike) -> Qt:
        if isinstance(x, UnitTracer) and x._trace is self:
            return x._q

        if isinstance(x, Qt) and isinstance(x._val, UnitTracer):
            assert x._val._trace is not self

            # from ._rules import mul_units

            # inner_q = x._val._q
            # # TODO: is this always true?
            # assert inner_q._unit in [dimensionless, anyunit]
            # return quantity(inner_q._val, mul_units(inner_q._unit, x._unit))

        return Qt._create(x)

    def _process_primitive(
        self, primitive: Primitive, tracers: Sequence[ArrayLike], params
    ) -> tuple[ArrayLike, ...] | ArrayLike:
        # if primitive is pjit.pjit_p:
        #     with core.set_current_trace(self):
        #         return tuple(_debug_pjit_direct(*tracers, **params))

        with core.set_current_trace(self._parent):
            args = [self.ensure_quantity(x) for x in tracers]

            # this optimization is not valid for
            # make_unit_p, value_and_unit_p
            # and operations invloving jaxpr that can contain those
            # if all(x._unit == dimensionless for x in args):
            #     return primitive.bind(*(x._val for x in args), **params)

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

            extra_msg = pp.text("")
            ex_type = PintaxRuntimeError

            if isinstance(ex, PintaxError_forward):
                ex_type = ex.ex_type
                if ex.msg is not None:
                    extra_msg = pp.group(ex.msg)
                ex = None
            else:
                # extra_msg = pp.text(f"{type(ex)}")
                try:
                    extra_msg = pp_join(
                        pp_join(
                            "(",
                            pp.color(
                                pp.text(type(ex).__name__),
                                foreground=pp.Color.RED,
                            ),
                            ")",
                            sep="",
                        ),
                        *(str(ex).splitlines()),
                    )
                except Exception as fmt_ex:
                    extra_msg = pp.text(
                        f"<{fmt_ex} while formatting exception that caused this>"
                    )

            parts = [
                pp_nested(
                    pp.text(f"failed to process primitive {primitive}:"),
                    extra_msg,
                ),
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
            desc_fmt = desc.format()
            raise ex_type(desc_fmt) from ex


rules = ruleset[Unit]()
rules_complex = ruleset[Qt]()


@contextlib.contextmanager
def with_unit_trace():
    with core.take_current_trace() as parent:
        assert parent is not None
        trace = UnitTrace(parent)
        with core.set_current_trace(trace, check_leaks=True):
            yield trace


# @overload
# def unitify[**P](
#     fun: Callable[P, Array],
#     *,
#     unwrap_outs: Literal[True] = True,
#     force_dimensionless_outs: Literal[False] = False,
# ) -> Callable[P, Array | Qt]: ...


# @overload
# def unitify[**P](
#     fun: Callable[P, Array],
#     *,
#     unwrap_outs: Literal[False],
#     force_dimensionless_outs: Literal[False] = False,
# ) -> Callable[P, Qt]: ...


# @overload
# def unitify[**P](
#     fun: Callable[P, Array],
#     *,
#     unwrap_outs: Literal[True] = True,
#     force_dimensionless_outs: Literal[True],
# ) -> Callable[P, Array]: ...


# @overload
# def unitify[**P, R](
#     fun: Callable[P, R],
#     *,
#     unwrap_outs=True,
#     force_dimensionless_outs=False,
# ) -> Callable[P, R]: ...


# def unitify(
#     fun: Callable, *, unwrap_outs=True, force_dimensionless_outs=False
# ) -> Callable:

#     assert False
#     # """
#     # first 3 overloads extend to pytrees
#     # last overload infer a generic type that might not be correct
#     # """
#     # if force_dimensionless_outs and not unwrap_outs:
#     #     raise TypeError()

#     # def _unitify_flat(
#     #     ctx: flattenctx, bufs: Sequence[QuantityLike]
#     # ) -> Sequence[Qt | ArrayLike]:
#     #     with with_unit_trace() as trace:
#     #         bufs_q = [UnitTracer(trace, trace.ensure_quantity(x)) for x in bufs]
#     #         out_bufs = ctx.call(bufs_q)
#     #         out_bufs_q = [trace.ensure_quantity(x) for x in out_bufs]
#     #         if unwrap_outs:
#     #             ans = [x._val if x._unit == dimensionless else x for x in out_bufs_q]
#     #         elif force_dimensionless_outs:
#     #             for x in out_bufs_q:
#     #                 assert x._unit == dimensionless
#     #             ans = [x._val for x in out_bufs_q]
#     #         else:
#     #             ans = out_bufs_q

#     #         # for x in ans:
#     #         #     if isinstance(x, Quantity):
#     #         #         if isinstance(x._val, UnitTracer) and x._val.trace is trace:
#     #         #             raise TypeError()
#     #         #         if isinstance(x._val, Quantity):
#     #         #             raise TypeError()

#     #         return ans

#     # return with_flatten(
#     #     fun,
#     #     _unitify_flat,
#     #     lambda arg: isinstance(arg, QuantityLike),
#     # )


def _debug_pjit_direct(*args, jaxpr: core.ClosedJaxpr, **_):
    assert len(jaxpr.consts) == 0
    return core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
