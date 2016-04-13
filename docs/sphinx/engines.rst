CrossCat Engines
================

A CrossCat engine can be a LocalEngine or a MultiprocessingEngine.  The CrossCatClient class can be used to dispatch importing and constructing the desired engine type.

.. warning::

   LocalEngine can only be used after the compilation of Cython wrapped C++ engine.

.. warning::

   The user should verify the existence of a remote engine before trying to use it.

Drawing samples
~~~~~~~~~~~~~~~
CrossCat works by exploring the space of possible partitionings of all the columns, and the partitioning of all the rows within each of the column partitions.  It runs Markov Chain Monte Carlo transition operators on the column and row partitionings, the column and row paritioning hyperparameters, and the column component model hyperparameters, to draw samples from the posterior distribution of partitionings.  These samples from the posterior are equivalent to estimates of the full joint distriubtion on columns and are inputs to functions that perform operations with the full joint distriubtion.

The two operations requried to draw samples from the posterior distribution are initialize and analyze.  The following is an example of instantiating a local engine object and drawing a sample from the posterior, given a data file.
 
   .. literalinclude:: simple_engine_example.py

.. warning::

   Here, we transition the Markov Chain only a few times so that it runs quickly on a small dataset.  A good rule of thumb is that it takes > 100 transitions to converge to the true distribution

Workings with samples
~~~~~~~~~~~~~~~~~~~~~

Once you have samples from the posterior, engines provide functionality to impute missing values in observed rows, sample values from unobserved rows, and determine probabilities of values in both observed and unobserved rows.

LocalEngine
~~~~~~~~~~~
.. automodule:: crosscat.LocalEngine
   :members:
   :undoc-members:
   :private-members:
   :show-inheritance:

MultiprocessingEngine
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: crosscat.MultiprocessingEngine
   :members:
   :undoc-members:
   :private-members:
   :show-inheritance:

CrossCatClient
~~~~~~~~~~~~~~
.. automodule:: crosscat.CrossCatClient
   :members:
   :undoc-members:
   :private-members:
   :show-inheritance:
