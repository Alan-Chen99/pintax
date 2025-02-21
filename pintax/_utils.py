import functools
from dataclasses import dataclass
from typing import Any, Callable, Concatenate, Sequence

import jax
import jax._src.pretty_printer as pp
import jax.tree_util as jtu
import numpy as np
from equinox._pretty_print import tree_pp
from jax import core
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
    def __init__(self, _: T):
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


@dataclass
class objwrapper:
    obj: Any


# for IPython autoreload
class allow_autoreload(type):
    def __new__(cls, func):
        del func
        return super().__new__(cls, "", (), {})

    def __init__(self, func):
        self.func = objwrapper(func)
        self.__module__ = func.__module__

    @property
    def __call__(self):  # type: ignore
        try:
            return self.func.obj
        except AttributeError:
            # inspect.signature(self) goes here
            return None

    def __get__(self, obj, objtype=None):
        del objtype
        if obj is None:
            return self
        return self.func.obj.__get__(obj)

    def __repr__(self):
        return repr(self.func.obj)


def compose_fn[**P, T, A](f1: Callable[P, T], f2: Callable[[T], A]) -> Callable[P, A]:
    def inner(*args: P.args, **kwargs: P.kwargs):
        t = f1(*args, **kwargs)
        return f2(t)

    return inner


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

                def inner(trace, *args, **kwargs):
                    return handle_res(f(trace, *args, **kwargs))

                self[prim] = inner
                return f

            return _setter1

        else:

            def _setter2(f: Callable[..., R | Sequence[R]]):
                assert prim not in self

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
class flattenctx:
    f: Callable[..., Any]
    is_leaf: is_leaf_t
    in_tree: jtu.PyTreeDef

    _out_store: lu.Store

    @staticmethod
    def create(f: Callable[..., Any], args, kwargs, is_leaf: is_leaf_t = None):
        args_flat, in_tree = jtu.tree_flatten((args, kwargs), is_leaf)
        return flattenctx(f, is_leaf, in_tree, lu.Store()), args_flat

    def call(self, args_flat_trans: Sequence):
        args_trans, kwargs_trans = jtu.tree_unflatten(self.in_tree, args_flat_trans)
        ans = self.f(*args_trans, **kwargs_trans)
        out_bufs, out_tree = jtu.tree_flatten(ans, self.is_leaf)
        self._out_store.store(out_tree)
        return tuple(out_bufs)

    @property
    def out_tree(self) -> jtu.PyTreeDef:
        return self._out_store.val  # type: ignore


def with_flatten[**P, T](
    f: Callable[Concatenate[P], T],
    handle_flat: Callable[[flattenctx, Sequence], Sequence],
    is_leaf: Callable[[Any], bool] | None = None,
) -> Callable[P, T]:

    @allow_autoreload
    @functools.wraps(f)
    @api_boundary
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        fctx, args_flat = flattenctx.create(f, args, kwargs, is_leaf)
        out_bufs_trans = handle_flat(fctx, args_flat)
        return jtu.tree_unflatten(fctx.out_tree, out_bufs_trans)

    return wrapped  # type: ignore


def check_unit(x) -> Unit:
    assert isinstance(x, Unit)
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
    if isinstance(x, core.Tracer):
        return x._pretty_print()
    return tree_pp(
        x,
        indent=2,
        short_arrays=False,
        struct_as_array=False,
        follow_wrapped=True,
        truncate_leaf=lambda _: False,
    )


def _pp_doc(x: pp.Doc | str) -> pp.Doc:
    if isinstance(x, pp.Doc):
        return x
    return pp.text(x)


def pp_nested(*docs: pp.Doc | str) -> pp.Doc:
    return pp.group(pp.nest(2, pp.join(pp.brk(), [_pp_doc(x) for x in docs])))


def check_arraylike(x: ArrayLike):
    core.get_aval(x)
