from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence

from jax import Array, lax
from jax import tree_util as jtu
from jax._src import core
from jax._src.callback import io_callback_p, pure_callback_p
from jax._src.core import Primitive
from jax._src.custom_batching import custom_vmap_p
from jax._src.debugging import debug_callback_p
from jax._src.lax.control_flow.solves import _LinearSolveTuple, _split_linear_solve_args
from jax._src.typing import ArrayLike

from ._core import (
    PintaxError_forward,
    Qt,
    Unit,
    UnitTrace,
    UnitTracer,
    anyunit,
    dimensionless,
    inv_unit,
    quantity,
    rules_complex,
    with_unit_trace,
)
from ._rules import dimensionless_or_err, sync_dims, unitify_jaxpr
from ._utils import (
    cast,
    cast_unchecked,
)
from ._utils import safe_zip as zip
from ._utils import (
    tree_map,
    weakref_lru_cache_,
)

# traceback_util.register_exclusion(__file__)


@dataclass(frozen=True)
class wrapped_callback[R]:
    callback: Callable[..., R]
    units: tuple[Unit, ...]

    def __call__(self, *args: ArrayLike) -> R:
        @with_unit_trace
        def inner(trace: UnitTrace):
            args_ = [
                UnitTracer(trace, quantity(x, u)) for x, u in zip(args, self.units)
            ]
            ans = self.callback(*args_)
            try:
                return tree_map(
                    lambda x: trace.handle_primitive_arg(x)._to(dimensionless)._val, ans
                )
            except PintaxError_forward as e:
                raise e.unwrap() from None

        return inner()


@weakref_lru_cache_
def wrap_callback[R](
    callback: Callable[..., R], units: tuple[Unit, ...]
) -> wrapped_callback[R]:
    return wrapped_callback(callback, units)


@rules_complex.many([debug_callback_p, pure_callback_p, io_callback_p])
def _(prim: Primitive):
    def func(*args: Qt, callback: Callable[..., Sequence[Array]], **kwargs):
        new_callback = wrap_callback(callback, tuple(x._unit for x in args))
        ans: Sequence[Array] = prim.bind(
            *[x._val for x in args], callback=new_callback, **kwargs
        )
        return tuple(quantity(x, dimensionless) for x in ans)

    return func


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


def _tup_units(qs: tuple[Qt, ...]) -> tuple[Unit, ...]:
    return tuple(x._unit for x in qs)


def _tup_vals(qs: tuple[Qt, ...]) -> tuple[ArrayLike, ...]:
    return tuple(x._val for x in qs)


@rules_complex(lax.while_p)
def _(
    *args: Qt,
    cond_jaxpr: core.ClosedJaxpr,
    body_jaxpr: core.ClosedJaxpr,
    body_nconsts: int,
    cond_nconsts: int,
):
    assert len(cond_jaxpr.consts) == 0
    assert len(body_jaxpr.consts) == 0

    # cond_nconsts=len(cond_consts)
    # body_nconsts=len(body_consts)
    # args = (*cond_consts, *body_consts, *init_vals)

    cond_consts = args[:cond_nconsts]
    body_consts = args[cond_nconsts : cond_nconsts + body_nconsts]
    init_vals = args[cond_nconsts + body_nconsts :]

    def get_state_units() -> tuple[Unit, ...]:
        state_units = _tup_units(init_vals)
        for _ in range(1000):
            _, new_units = unitify_jaxpr(
                body_jaxpr.jaxpr,
                _tup_units(body_consts) + state_units,
            )
            if not any(
                x == anyunit and y != anyunit for x, y in zip(state_units, new_units)
            ):
                return new_units
            state_units = new_units

        raise RuntimeError("infinite loop")

    state_units = get_state_units()

    new_cond, _ = unitify_jaxpr(
        cond_jaxpr.jaxpr,
        _tup_units(cond_consts) + state_units,
        force_out_units=(dimensionless,),
    )
    new_body, _ = unitify_jaxpr(
        body_jaxpr.jaxpr,
        _tup_units(body_consts) + state_units,
        force_out_units=state_units,
    )

    out_vals: tuple[ArrayLike] = lax.while_p.bind(
        *_tup_vals(cond_consts),
        *_tup_vals(body_consts),
        *[x._to(u)._val for x, u in zip(init_vals, state_units)],
        cond_jaxpr=core.ClosedJaxpr(new_cond, cond_jaxpr.consts),
        body_jaxpr=core.ClosedJaxpr(new_body, body_jaxpr.consts),
        body_nconsts=body_nconsts,
        cond_nconsts=cond_nconsts,
    )
    return tuple(quantity(x, u) for x, u in zip(out_vals, state_units))


@rules_complex(lax.cond_p)
def _(which: Qt, *args: Qt, branches: tuple[core.ClosedJaxpr, ...]):
    dimensionless_or_err(which._unit)

    for x in branches:
        assert len(x.consts) == 0

    out_units = [
        unitify_jaxpr(jaxpr.jaxpr, tuple(x._unit for x in args))[1]
        for jaxpr in branches
    ]

    out_units_cast_to = tuple(
        sync_dims(*(quantity(1.0, z) for z in x))[0] for x in zip(*out_units)
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


@rules_complex(custom_vmap_p)
def _(*args: Qt, call: core.ClosedJaxpr, **kwargs):

    assert isinstance(call, core.ClosedJaxpr)

    if len(call.jaxpr.constvars) > 0:
        raise NotImplementedError()

    out_jaxpr, out_units = unitify_jaxpr(
        call.jaxpr,
        tuple(x._unit for x in args),
    )

    out_vals: tuple[ArrayLike] = custom_vmap_p.bind(
        *(x._val for x in args),
        call=core.ClosedJaxpr(out_jaxpr, call.consts),
        **kwargs,
    )

    return tuple(quantity(x, u) for x, u in zip(out_vals, out_units))


class LinearSolveTuple[T](NamedTuple):
    matvec: T
    vecmat: T
    solve: T
    transpose_solve: T


@rules_complex(lax.linear_solve_p)
def _(
    *args: Qt,
    const_lengths: LinearSolveTuple[int],
    jaxprs: LinearSolveTuple[core.ClosedJaxpr],
):
    for x in jaxprs:
        assert isinstance(x, core.ClosedJaxpr)
        assert len(x.consts) == 0

    consts_, b_ = _split_linear_solve_args(args, const_lengths)
    consts = cast_unchecked[LinearSolveTuple[list[Qt]]]()(consts_)
    b = cast[list[Qt]]()(b_)

    b_units = tuple(x._unit for x in b)
    new_solve, a_units = unitify_jaxpr(
        jaxprs.solve.jaxpr,
        tuple(x._unit for x in consts.solve) + b_units,
    )

    new_matvec, _ = unitify_jaxpr(
        jaxprs.matvec.jaxpr,
        tuple(x._unit for x in consts.matvec) + a_units,
        force_out_units=b_units,
    )

    a_units_inv = tuple(inv_unit(x) for x in a_units)
    b_units_inv = tuple(inv_unit(x) for x in b_units)

    new_vecmat, _ = unitify_jaxpr(
        jaxprs.vecmat.jaxpr,
        tuple(x._unit for x in consts.vecmat) + b_units_inv,
        force_out_units=a_units_inv,
    )
    new_transpose_solve, _ = unitify_jaxpr(
        jaxprs.transpose_solve.jaxpr,
        tuple(x._unit for x in consts.transpose_solve) + a_units_inv,
        force_out_units=b_units_inv,
    )

    new_jaxprs = LinearSolveTuple[core.Jaxpr](
        matvec=new_matvec,
        vecmat=new_vecmat,
        solve=new_solve,
        transpose_solve=new_transpose_solve,
    )

    a_outs = lax.linear_solve_p.bind(
        *(x._val for x in args),
        const_lengths=const_lengths,
        jaxprs=_LinearSolveTuple(*(core.ClosedJaxpr(x, []) for x in new_jaxprs)),
    )

    return tuple(quantity(x, u) for x, u in zip(a_outs, a_units))
