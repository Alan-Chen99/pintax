from __future__ import annotations

import contextlib
import functools
from functools import partial
from typing import Callable, Literal, Sequence, final, overload

import equinox as eqx
import jax._src.pretty_printer as pp
import numpy as np
import pint
from jax import Array, lax
from jax import numpy as jnp
from jax._src import ad_util, core, traceback_util
from jax._src.core import Primitive
from jax._src.interpreters import ad, batching
from jax._src.numpy.util import promote_args
from jax._src.traceback_util import api_boundary
from jax._src.typing import ArrayLike, DType, Shape
from jax.interpreters import mlir
from pint import Unit, UnitRegistry
from pint.errors import PintTypeError

from ._utils import (
    cast_unchecked,
    check_arraylike,
    check_unit,
    dict_set,
    dtype_of,
    flattenctx,
    jit,
    pp_nested,
    pretty_print,
    ruleset,
    with_flatten,
)

traceback_util.register_exclusion(__file__)

_global_ureg = UnitRegistry()
dimensionless = _global_ureg.dimensionless
_global_ureg.define("pintax_symbolic_zero=[pintax_symbolic_zero]")
symbolic_zero: Unit = _global_ureg.pintax_symbolic_zero


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


def _unitify_check_ret(x: Quantity, trace: UnitTrace) -> Quantity:
    if isinstance(x._val, UnitTracer) and x._val._trace is trace:
        raise TypeError()
    if isinstance(x._val, Quantity):
        raise TypeError()
    return x


def _quantity_binop(
    fun: Callable[[Array, Array], Array], reverse=False
) -> Callable[[QuantityLike, QuantityLike], Quantity]:

    @functools.wraps(fun)
    @api_boundary
    def inner(x1: QuantityLike, x2: QuantityLike, /) -> Quantity:
        if reverse:
            x1, x2 = x2, x1
        for x in [x1, x2]:
            if isinstance(x, Array):
                raise TypeError(f"expected non-jax constant, got\n{x}")
        # if any(isinstance(x, Array) for x in (x1, x2)):
        #     raise TypeError()
        #     # x1, x2 = (Quantity._create(x).as_array() for x in (x1, x2))
        #     # x1, x2 = promote_args(str(fun), x1, x2)
        #     # return fun(x1, x2)
        with with_unit_trace() as trace:
            x1, x2 = (UnitTracer(trace, trace.ensure_quantity(x)) for x in (x1, x2))
            x1, x2 = promote_args(str(fun), x1, x2)
            ans = fun(x1, x2)
            return trace.ensure_quantity(ans)

    return inner


@final
class Quantity(eqx.Module):
    # _val is allowed to be a UnitTracer
    # _val is never allowed to be a Quantity
    # _unit must always be a Unit
    _val: ArrayLike
    _unit: Unit = eqx.field(static=True)

    def __init__(self, *, _val: ArrayLike, _unit: Unit):
        assert isinstance(_unit, Unit)
        assert not isinstance(_val, Quantity)
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

    def _to(self, new_unit: Unit) -> Quantity:
        if self._unit == new_unit:
            return self
        if self._unit == symbolic_zero:
            return quantity(self._val, new_unit)

        return self._to_slow(new_unit)

    @partial(jit, static_argnames=["new_unit"], inline=True)
    def _to_slow(self, new_unit: Unit) -> Quantity:
        dtype = dtype_of(self._val)
        assert not np.issubdtype(dtype, np.integer)
        ans = _global_ureg.Quantity(self._val, self._unit).to(new_unit)
        assert ans.units == new_unit
        assert isinstance(ans.magnitude, Array)
        if ans.magnitude.dtype == dtype:
            out_mag = ans.magnitude
        else:
            out_mag = lax.convert_element_type(ans.magnitude, dtype)
        return quantity(out_mag, new_unit)

    @staticmethod
    def _create(v: QuantityLike) -> Quantity:
        if isinstance(v, Quantity):
            return v
        if isinstance(v, pint.Quantity):
            unit = v.units
            assert isinstance(unit, Unit)
            check_arraylike(v.magnitude)
            return quantity(v.magnitude, unit)
        if isinstance(v, Unit):
            return quantity(1.0, v)

        if isinstance(v, int | float) and v == 0.0:
            return quantity(v, symbolic_zero)

        if isinstance(v, np.ndarray) and v.shape == () and v == 0.0:
            return quantity(v, symbolic_zero)

        check_arraylike(v)
        return quantity(v, dimensionless)

    def _pretty_print(self, prefix: pp.Doc = pp.text("Quantity")) -> pp.Doc:
        # modified from equinox._pretty_print
        val = pretty_print(self._val)
        unit = pp.color(
            pp.text(repr(str(self._unit))),
            foreground=pp.Color.BLUE,
            intensity=pp.Intensity.BRIGHT,
        )

        _comma_sep = pp.concat([pp.text(","), pp.brk()])
        nested = pp.concat(
            [
                pp.nest(2, pp.concat([pp.brk(""), pp.join(_comma_sep, [val, unit])])),
                pp.brk(""),
            ]
        )
        return pp.group(pp.concat([prefix, pp.text("("), nested, pp.text(")")]))

    def __repr__(self):
        return self._pretty_print().format()

    @api_boundary
    def as_array(self) -> Array:
        ans = self._val * make_unit(self._unit)
        assert isinstance(ans, Array)
        return ans

    __jax_array__ = as_array

    # lax does not jit and gives better unit errors
    __add__ = _quantity_binop(lax.add)
    __radd__ = _quantity_binop(lax.add, reverse=True)

    __sub__ = _quantity_binop(lax.sub)
    __rsub__ = _quantity_binop(lax.sub, reverse=True)

    __mul__ = _quantity_binop(lax.mul)
    __rmul__ = _quantity_binop(lax.mul, reverse=True)

    __truediv__ = _quantity_binop(lax.div)
    __rtruediv__ = _quantity_binop(lax.div, reverse=True)

    def __neg__(self) -> Quantity:
        return quantity(-cast_unchecked[Array]()(self._val), self._unit)

    def __pow__(self, p: int | float) -> Quantity:
        assert isinstance(p, int | float)
        return quantity(self._val**p, check_unit(self._unit**p))

    def __float__(self):
        return float(cast_unchecked()(self._to(dimensionless)._val))

    def __array__(self):
        return np.array(cast_unchecked()(self._to(dimensionless)._val))


def quantity(val: ArrayLike, unit: Unit):
    return Quantity(_val=val, _unit=unit)


QuantityLike = ArrayLike | Quantity | pint.Quantity | Unit


class UnitTracer(core.Tracer):
    _q: Quantity

    def __init__(self, trace: core.Trace, val: Quantity):
        assert isinstance(val, Quantity)
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
        return self._q.__array__()


class UnitTrace(core.Trace[UnitTracer]):

    def __init__(self, parent: core.Trace):
        self._parent = parent

    def invalidate(self):
        print("invalidate:", self, id(self))
        super().invalidate()

    def ensure_quantity(self, x: QuantityLike) -> Quantity:
        if isinstance(x, UnitTracer) and x._trace is self:
            return x._q

        # this is allowed; for ex with
        # unitify(lambda: qreg.meter + qreg.meter)()
        if (
            isinstance(x, Quantity)
            and isinstance(x._val, UnitTracer)
            and x._val._trace is self
        ):
            from ._rules import mul_units

            inner_q = x._val._q
            # TODO: is this always true?
            assert inner_q._unit in [dimensionless, symbolic_zero]
            return quantity(inner_q._val, mul_units(inner_q._unit, x._unit))

        return Quantity._create(x)

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

            if primitive in _rules_complex:
                out_quants = _rules_complex[primitive](self, *args, **params)
            elif primitive in _rules:
                out_units = _rules[primitive](self, *(x._unit for x in args), **params)

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
                    extra_msg = pp.join(
                        pp.brk(), [pp.text(f"({type(ex).__name__})"), pp.text(str(ex))]
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
            raise ex_type(desc.format()) from ex


_rules = ruleset[Unit]()
_rules_complex = ruleset[Quantity]()


@overload
def unitify[**P](
    fun: Callable[P, Array],
    *,
    unwrap_outs: Literal[True] = True,
    force_dimensionless_outs: Literal[False] = False,
) -> Callable[P, Array | Quantity]: ...


@overload
def unitify[**P](
    fun: Callable[P, Array],
    *,
    unwrap_outs: Literal[False],
    force_dimensionless_outs: Literal[False] = False,
) -> Callable[P, Quantity]: ...


@overload
def unitify[**P](
    fun: Callable[P, Array],
    *,
    unwrap_outs: Literal[True] = True,
    force_dimensionless_outs: Literal[True],
) -> Callable[P, Array]: ...


@overload
def unitify[**P, R](
    fun: Callable[P, R],
    *,
    unwrap_outs=True,
    force_dimensionless_outs=False,
) -> Callable[P, R]: ...


@contextlib.contextmanager
def with_unit_trace():
    with core.take_current_trace() as parent:
        assert parent is not None
        trace = UnitTrace(parent)
        with core.set_current_trace(trace, check_leaks=True):
            yield trace


def unitify(
    fun: Callable, *, unwrap_outs=True, force_dimensionless_outs=False
) -> Callable:
    """
    first 3 overloads extend to pytrees
    last overload infer a generic type that might not be correct
    """
    if force_dimensionless_outs and not unwrap_outs:
        raise TypeError()

    def _unitify_flat(
        ctx: flattenctx, bufs: Sequence[QuantityLike]
    ) -> Sequence[Quantity | ArrayLike]:
        with with_unit_trace() as trace:
            bufs_q = [UnitTracer(trace, trace.ensure_quantity(x)) for x in bufs]
            out_bufs = ctx.call(bufs_q)
            out_bufs_q = [trace.ensure_quantity(x) for x in out_bufs]
            if unwrap_outs:
                ans = [x._val if x._unit == dimensionless else x for x in out_bufs_q]
            elif force_dimensionless_outs:
                for x in out_bufs_q:
                    assert x._unit == dimensionless
                ans = [x._val for x in out_bufs_q]
            else:
                ans = out_bufs_q

            # for x in ans:
            #     if isinstance(x, Quantity):
            #         if isinstance(x._val, UnitTracer) and x._val.trace is trace:
            #             raise TypeError()
            #         if isinstance(x._val, Quantity):
            #             raise TypeError()

            return ans

    return with_flatten(
        fun,
        _unitify_flat,
        lambda arg: isinstance(arg, QuantityLike),
    )


def _debug_pjit_direct(*args, jaxpr: core.ClosedJaxpr, **_):
    assert len(jaxpr.consts) == 0
    return core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)


@api_boundary
def make_unit(unit: Unit) -> Array:
    assert isinstance(unit, Unit)
    ans = make_unit_p.bind(unit=unit)
    assert isinstance(ans, Array)
    return ans


@api_boundary
def value_and_unit(value: ArrayLike) -> tuple[Array, Array]:
    val, unit = value_and_unit_p.bind(value)
    assert isinstance(val, Array)
    assert isinstance(unit, Array)
    return val, unit


# make_unit_p
make_unit_p = core.Primitive("make_unit")


@make_unit_p.def_abstract_eval
def _(*, unit: Unit):
    assert isinstance(unit, Unit)
    return core.get_aval(1.0)


@_rules_complex(make_unit_p)
def _(*, unit: Unit):
    return quantity(1.0, unit)


@make_unit_p.def_impl
def _(*, unit: Unit):
    raise TypeError(f"trying to use {unit!r} outside of unitify")


@partial(mlir.register_lowering, make_unit_p)
@cast_unchecked[mlir.LoweringRule]()
def _(ctx, *, unit: Unit):
    raise TypeError(f"trying to use {unit!r} outside of unitify")


# value_and_unit_p
value_and_unit_p = core.Primitive("value_and_unit")
value_and_unit_p.multiple_results = True


@value_and_unit_p.def_abstract_eval
def _(arg: core.ShapedArray):
    return arg, core.ShapedArray(shape=(), dtype=jnp.int32)


@_rules_complex(value_and_unit_p)
def _(x: Quantity):
    return quantity(x._val, dimensionless), quantity(1.0, x._unit)


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
