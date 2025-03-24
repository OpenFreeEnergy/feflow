=================
feflow Change Log
=================

.. current developments

v0.1.2
====================

- Support for ``openfe >=1.1.0``. Importing ``openmm`` utils from the correct modules for recent ``openfe`` versions (`PR #109 <https://github.com/OpenFreeEnergy/feflow/pull/109>`_).
- Support for recent ``pymbar`` versions (``>4.0``). Dropped support for ``pymbar`` 3 or previous versions (`PR #109 <https://github.com/OpenFreeEnergy/feflow/pull/109>`_).

v0.1.1
====================

- Support for ``pydantic`` >=1.10.17, allowing ``pydantic`` 2 to coexist in the same environment (`PR #58 <https://github.com/OpenFreeEnergy/feflow/pull/58>`_).
- Minimization is now performed in the ``CycleUnit`` instead of the ``SetupUnit`` (`PR #60 <https://github.com/OpenFreeEnergy/feflow/pull/60>`_).
- Added protein-ligand complex testing (`PR #61 <https://github.com/OpenFreeEnergy/feflow/pull/61>`_).
- Added GPU Continuous Integration (CI) (`PR #64 <https://github.com/OpenFreeEnergy/feflow/pull/64>`_).

v0.1
====================

First release. Base implementation of Nonequilibrium cycling protocol.

