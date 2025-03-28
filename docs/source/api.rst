.. currentmodule:: pintax

#####
 API
#####

.. autofunction:: unitify

.. py:data:: areg

   .. autofunction:: pintax.areg.__getattr__

   .. autofunction:: pintax.areg.__call__

.. autoclass:: QuantityLike

.. autofunction:: quantity

.. autoclass:: Quantity
   :exclude-members: __init__, __new__

   .. autoattribute:: m

   .. autoproperty:: u

   .. autoproperty:: a

   .. automethod:: to

   .. autoproperty:: shape

   .. autoproperty:: dtype

   .. autoproperty:: size

   .. autoproperty:: ndim

.. autoclass:: Unit
   :exclude-members: __init__, __new__

   .. autoproperty:: a

   .. automethod:: __jax_array__

.. py:data:: ureg

   .. autofunction:: pintax.ureg.__getattr__

   .. autofunction:: pintax.ureg.__call__

.. autofunction:: convert_unit

.. autofunction:: magnitude

.. autofunction:: sync_units
