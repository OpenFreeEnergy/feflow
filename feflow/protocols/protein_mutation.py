"""
Implementation of protein mutation protocol based on Nonequilibrium Cycling, using OpenMM as
MD engine.
"""
# TODO: WE might not need a whole new Protocol for protein mutations after all
from feflow.protocols import NonEquilibriumCyclingProtocol


ProteinMutationProtocol = NonEquilibriumCyclingProtocol
