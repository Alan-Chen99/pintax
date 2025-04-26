import functools
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Concatenate, Never

import jax
import jax._src.pretty_printer as pp
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax import tree_util as jtu
from jax._src import core, traceback_util
from jax._src.typing import ArrayLike, DType, Shape
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


def cast_unchecked_(x):
    return cast_unchecked()(x)


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

    def many(self, prims: Iterable[core.Primitive], trace: bool = False):
        def _setter(f: Callable[[core.Primitive], Callable[..., R | Sequence[R]]]):
            for prim in prims:
                self(prim, trace=trace)(f(prim))
            return f

        return _setter


def check_unit(x) -> Unit:
    if not isinstance(x, Unit):
        raise TypeError(f"expected pint.Unit, got {x} ({type(x)})")
    return x


def weakref_lru_cache_[F: Callable](f: F) -> F:
    return cast_unchecked_(weakref_lru_cache(f))


def _wrap_jit[F: Callable, **P](
    jit_fn: Callable[Concatenate[Callable, P], Any],
) -> Callable[Concatenate[F, P], F]:
    return jit_fn  # type: ignore


jit = _wrap_jit(jax.jit)


def dtype_of(x: ArrayLike | core.AbstractValue) -> np.dtype:
    dtype = core.get_aval(x).dtype  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(dtype, np.dtype)
    return dtype


def shape_of(x: ArrayLike | core.AbstractValue) -> Shape:
    if isinstance(x, core.AbstractValue):
        return x.shape  # pyright: ignore[reportAttributeAccessIssue]
    return jnp.shape(x)


def dict_set[K, V](d: dict[K, V], k: K) -> Callable[[V], V]:
    def inner(v: V):
        d[k] = v
        return v

    return inner


def pretty_print(x: Any) -> pp.Doc:
    # if isinstance(x, core.Tracer):
    #     return x._pretty_print()
    ans = repr(x)
    return pp_join(*ans.splitlines(), sep=pp.brk(" " * 100))


def _pp_doc(x: pp.Doc | str) -> pp.Doc:
    if isinstance(x, pp.Doc):
        return x
    return pp.text(x)


def pp_join(*docs: pp.Doc | str, sep: pp.Doc | str | None = None) -> pp.Doc:
    if sep is None:
        sep = pp.brk()
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
    x = cast_unchecked()(x)
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


def property_method[S, **P, R](
    prop: Callable[[S], Callable[P, R]],
):
    def inner(self: S, *args: P.args, **kwargs: P.kwargs) -> R:
        return prop(self)(*args, **kwargs)

    return inner


def _tree_map[T](
    f: Callable, tree: T, *rest: T, is_leaf: Callable[[Any], bool] | None = None
) -> T:
    assert False


tree_map = cast_unchecked(_tree_map)(jtu.tree_map)
