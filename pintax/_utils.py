import functools
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Concatenate, Never, Sequence, TypeGuard

import jax
import jax._src.pretty_printer as pp
import jax.tree_util as jtu
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax._src import core
from jax._src import linear_util as lu
from jax._src import traceback_util
from jax._src.traceback_util import api_boundary
from jax._src.typing import ArrayLike
from jax._src.util import weakref_lru_cache
from pint import Unit

traceback_util.register_exclusion(__file__)


class Empty_t:
    pass


_empty = Empty_t()


class cast_unchecked[T]:
    def __init__(self, _: T | Empty_t = _empty):
        pass

    def __call__(self, a) -> T:
        return a


class cast[T]:
    def __init__(self, _: T | Empty_t = _empty):
        pass

    def __call__(self, a: T) -> T:
        return a


@cast_unchecked(zip)
def safe_zip(*args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(zip(*args))


class ruleset[R](dict[core.Primitive, Callable[..., tuple[R, ...]]]):
    def __init__(self):
        super().__init__()

    def __call__(
        self, prim: core.Primitive, trace: bool = False
    ) -> Callable[[Callable[..., R | Sequence[R]]], Any]:
        def handle_res(v: R | Sequence[R]) -> tuple[R, ...]:
            if prim.multiple_results:
                assert isinstance(v, Sequence)
                return tuple(v)
            else:
                assert not isinstance(v, Sequence)
                return (v,)

        if trace:

            def _setter1(f: Callable[..., R | Sequence[R]]):
                assert prim not in self

                @functools.wraps(f)
                def inner(trace, *args, **kwargs):
                    return handle_res(f(trace, *args, **kwargs))

                self[prim] = inner
                return f

            return _setter1

        else:

            def _setter2(f: Callable[..., R | Sequence[R]]):
                assert prim not in self

                @functools.wraps(f)
                def inner(trace, *args, **kwargs):
                    del trace
                    return handle_res(f(*args, **kwargs))

                self[prim] = inner
                return f

            return _setter2

    def many(self, prims: Sequence[core.Primitive], trace: bool = False):
        def _setter(f: Callable[[core.Primitive], Callable[..., R | Sequence[R]]]):
            for prim in prims:
                self(prim, trace=trace)(f(prim))
            return f

        return _setter


is_leaf_t = Callable[[Any], bool] | None


@dataclass
class flattenctx[Leaf]:
    f: Callable[..., Any]
    is_leaf: Callable[[Any], TypeGuard[Leaf]]
    in_tree: jtu.PyTreeDef

    _out_store: lu.Store

    @staticmethod
    def create(
        f: Callable[..., Any], args, kwargs, is_leaf: Callable[[Any], TypeGuard[Leaf]]
    ):
        args_flat, in_tree = jtu.tree_flatten((args, kwargs), is_leaf)
        return flattenctx(f, is_leaf, in_tree, lu.Store()), args_flat

    def call(self, args_flat_trans: Sequence[ArrayLike]) -> tuple[Leaf, ...]:
        args_trans, kwargs_trans = jtu.tree_unflatten(self.in_tree, args_flat_trans)
        ans = self.f(*args_trans, **kwargs_trans)
        out_bufs, out_tree = jtu.tree_flatten(ans, self.is_leaf)
        self._out_store.store(out_tree)
        return tuple(out_bufs)

    @property
    def out_tree(self) -> jtu.PyTreeDef:
        return self._out_store.val  # type: ignore


def with_flatten[Leaf, **P, T](
    f: Callable[Concatenate[P], T],
    handle_flat: Callable[[flattenctx[Leaf], Sequence[Leaf]], Sequence[Leaf]],
    is_leaf: Callable[[Any], TypeGuard[Leaf]],
) -> Callable[P, T]:

    @functools.wraps(f)
    @api_boundary
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        fctx, args_flat = flattenctx.create(f, args, kwargs, is_leaf)
        out_bufs_trans = handle_flat(fctx, args_flat)
        return jtu.tree_unflatten(fctx.out_tree, out_bufs_trans)

    try:
        setattr(wrapped, "__signature__", inspect.signature(f))
    except:
        pass
    return wrapped  # type: ignore


def check_unit(x) -> Unit:
    if not isinstance(x, Unit):
        raise TypeError(f"expected pint.Unit, got {x} ({type(x)})")
    return x


def weakref_lru_cache_[F: Callable](f: F) -> F:
    return weakref_lru_cache(f)  # type: ignore


def _wrap_jit[F: Callable, **P](
    jit_fn: Callable[Concatenate[Callable, P], Any],
) -> Callable[Concatenate[F, P], F]:
    return jit_fn  # type: ignore


jit = _wrap_jit(jax.jit)


def dtype_of(x: ArrayLike) -> np.dtype:
    dtype = core.get_aval(x).dtype  # type: ignore
    assert isinstance(dtype, np.dtype)
    return dtype


def dict_set[K, V](d: dict[K, V], k: K) -> Callable[[V], V]:
    def inner(v: V):
        d[k] = v
        return v

    return inner


def pretty_print(x: Any) -> pp.Doc:
    # if isinstance(x, core.Tracer):
    #     return x._pretty_print()
    ans = repr(x)
    return pp_join(*ans.splitlines())


def _pp_doc(x: pp.Doc | str) -> pp.Doc:
    if isinstance(x, pp.Doc):
        return x
    return pp.text(x)


def pp_join(*docs: pp.Doc | str, sep: pp.Doc | str = pp.brk()) -> pp.Doc:
    return pp.join(_pp_doc(sep), [_pp_doc(x) for x in docs])


def pp_nested(*docs: pp.Doc | str) -> pp.Doc:
    return pp.group(pp.nest(2, pp_join(*docs)))


def pp_obj(name: pp.Doc | str, *fields: pp.Doc | str):
    # modified from equinox._pretty_print
    _comma_sep = pp.concat([pp.text(","), pp.brk()])
    nested = pp.concat(
        [
            pp.nest(
                2,
                pp.concat(
                    [pp.brk(""), pp.join(_comma_sep, [_pp_doc(x) for x in fields])]
                ),
            ),
            pp.brk(""),
        ]
    )
    return pp.group(pp.concat([_pp_doc(name), pp.text("("), nested, pp.text(")")]))


def check_arraylike(x: ArrayLike) -> ArrayLike:
    if isinstance(x, ArrayLike):
        return x
    x = jnp.array(x)
    assert isinstance(x, Array)
    return x


def unreachable(x: Never) -> Never:
    raise RuntimeError("unreachable", x)


def arraylike_to_float(x: ArrayLike) -> float:
    return float(cast_unchecked()(x))


def ensure_jax(x: ArrayLike) -> Array:
    if isinstance(x, Array):
        return x
    return jnp.array(x)
