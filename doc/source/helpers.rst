
Helpers
=======

ANN wrapping
------------

When working with models, it is very common to get a few steps when making a prediction.
For example a common scenario is: Normalize, feed through model, inverse normalization.
The wrapper helper function can call such a chain of methods.

Below is an example of the above steps wrapped into a method call ``ev(...)`` (short for evaluate).

.. literalinclude:: ../../examples/wrapper_demo.py

Application Programming Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pypr.helpers.wrappers
   :members:


Application Programming Interface
---------------------------------

.. automodule:: pypr.helpers
   :members:

