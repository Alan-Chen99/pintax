.. currentmodule:: pintax

######################
 Pintax documentation
######################

This library allows attaching a `Pint <https://pint.readthedocs.io>`_ to
`JAX <https://jax.readthedocs.io>`_ Arrays using JAX transforms. Pintax
incurs no additional runtime cost for jitted functions.

Compared to `jpu <https://github.com/dfm/jpu>`_, this library allows you
to use ``jax.numpy`` or ``jax.lax`` apis directly, and that you can
:func:`pintax.unitify` any existing function that makes use of
``jax.numpy`` or ``jax.lax`` apis.

This library uses internal JAX apis. You may need to pin JAX to the
version pinned in ``poetry.lock``.

#########
 Example
#########

.. code:: python

   from typing import reveal_type

   import numpy as np
   from jax import Array, lax
   from jax import numpy as jnp

   from pintax import areg, quantity, unitify


   @unitify
   def main():
       # use the array registry pintax.areg to obtain an array which is the unit with magnitude 1
       reveal_type(areg.m)  # Array
       print(areg.m)  # UnitTracer(1, 'meter')

       print(5 * areg.m)  # UnitTracer(Array(5, dtype=int32, weak_type=True), 'meter')
       print(5.0 * areg.m)  # UnitTracer(Array(5., dtype=float32, weak_type=True), 'meter')

       a1 = jnp.array([1.0, 2.0]) * areg.m
       a2 = jnp.array([3.0, 4.0]) * areg.inch
       reveal_type(a1)  # Array
       reveal_type(a2)  # Array

       print(jnp.dot(a1, a2))  # UnitTracer(Array(11., dtype=float32), 'inch * meter')
       print(jnp.array([a1, a2]))
       # UnitTracer(
       #   Array([[1.    , 2.    ],
       #          [0.0762, 0.1016]], dtype=float32),
       #   'meter'
       # )

       # An error will be thrown on a unit mismatch
       # _ = lax.add(a1, 2.0 * areg.s)
       # pintax.unstable.PintaxDimensionalityError: failed to process primitive add:
       #   Cannot convert from 'second' ([time]) to 'meter' ([length])
       # with args:
       #   UnitTracer(Array([1., 2.], dtype=float32), 'meter')
       #   UnitTracer(Array(2., dtype=float32, weak_type=True), 'second')

       # only dimensionless quantities are convertable to numpy arrays.
       print(np.array(a2 / areg.m))  # [0.0762 0.1016]
       # _ = np.array(a2)
       # pintax.unstable.PintaxDimensionalityError: Cannot convert from 'inch' ([length]) to 'dimensionless' (dimensionless)

       # use quantity to seperate an array into units and magnitudes.
       a2_q = quantity(a2)
       reveal_type(a2_q)  # Quantity
       print(a2_q)  # Quantity(Array([3., 4.], dtype=float32), 'inch')
       print(a2_q.m)  # UnitTracer(Array([3., 4.], dtype=float32), 'dimensionless')
       print(a2_q.u)  # Unit('inch')
       print(a2_q.a)  # UnitTracer(Array([3., 4.], dtype=float32), 'inch')

       return a2 / a2_q.u**2


   ans = main()
   # output of a unitified function is converted to Quantity
   print(ans)  # Quantity(Array([3., 4.], dtype=float32), '1 / inch')

################
 Pintax and jit
################

.. code:: python

   import jax
   from jax import numpy as jnp
   from pintax import areg, quantity, unitify

   # jit outside unitify
   @jax.jit
   @unitify
   def main():

       v = jnp.array([1.0, 2.0]) * areg.m

       # jit outside unitify: value is not known but unit is known
       # UnitTracer(Traced<ShapedArray(float32[2])>with<DynamicJaxprTrace>, 'meter')
       print(v)

       # use jax.debug.print to print the value
       # value: UnitTracer(Array([1., 2.], dtype=float32), 'meter')
       jax.debug.print("value: {}", v)

       # debug callback runs under unitify
       jax.debug.callback(lambda x: print("value2:", x * areg.m), v)

       # throws PintaxDimensionalityError immediately
       # _ = v + areg.s

       # jit inside unitify
       @jax.jit
       def inner():
           v2 = v * 2
           # v2 = v
           # jit inside unitify: unit is not known
           # units are traced and then checked later
           print("v2:", v2)  # Traced<ShapedArray(float32[2])>with<DynamicJaxprTrace>

           # UnitTracer(Array([2., 4.], dtype=float32), 'meter')
           jax.debug.print("v2_callback: {}", v2)

           q2 = quantity(v2)
           # Quantity(
           #   Traced<ShapedArray(float32[2])>with<DynamicJaxprTrace>,
           #   Traced<ShapedArray(int32[])>with<DynamicJaxprTrace>
           # )
           print("q2:", q2)

           v3 = q2.m / q2.u
           print("v3:", v3)  # Traced<ShapedArray(float32[2])>with<DynamicJaxprTrace>
           # v3_callback: UnitTracer(Array([2., 4.], dtype=float32), '1 / meter')
           jax.debug.print("v3_callback: {}", v3)

           # exception is thrown after inner has been traced
           # location of error is available via JaxStackTraceBeforeTransformation
           # _ = v3 + areg.s

           return v3

       v3 = inner()
       # units are known again but not values since we are still in the outer jit
       print("v3 out", v3)


   if __name__ == "__main__":
       main()

.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 2

   api
   unstable
