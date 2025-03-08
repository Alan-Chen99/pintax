from jax import numpy as jnp
from jax._src.numpy.linalg import EighResult
from jax._src.typing import ArrayLike

from ._primitives import value_and_unit
from ._utils import cast


def eigh(a: ArrayLike, UPLO: str | None = None, symmetrize_input: bool = True):
    a_val, a_unit = value_and_unit(a)
    ans = jnp.linalg.eigh(a_val, UPLO, symmetrize_input)
    return EighResult(
        eigenvalues=ans.eigenvalues * a_unit,
        eigenvectors=ans.eigenvectors,
    )


@cast(jnp.linalg.lstsq)
def lstsq(
    a: ArrayLike, b: ArrayLike, rcond: float | None = None, *, numpy_resid: bool = False
):
    a_val, a_unit = value_and_unit(a)
    x, resid, rank, s = jnp.linalg.lstsq(a_val, b, rcond, numpy_resid=numpy_resid)
    return x / a_unit, resid, rank, s * a_unit
