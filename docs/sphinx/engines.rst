CrossCat Engines
================

A CrossCat engine can be local or remote.  A remote engine may be a JSONRPC engine or a Hadoop engine.  A CrssCatClient can be used to dispatch importing and constructing the desired engine type.

.. warning::

   LocalEngine can only be used after the compilation of Cython wrapped C++ engine.

.. warning::

   The user should verify the existence of a remote engine before trying to use it.

The following is an example of instantiation and running of a local engine

   .. literalinclude:: simple_engine_example.py

.. automodule:: crosscat.LocalEngine
   :members:
   :private-members:

.. automodule:: crosscat.JSONRPCEngine
   :members:
   :private-members:

.. automodule:: crosscat.HadoopEngine
   :members:
   :private-members:


