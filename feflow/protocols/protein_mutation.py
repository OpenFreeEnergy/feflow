"""
Implementation of protein mutation protocol based on Nonequilibrium Cycling, using OpenMM as
MD engine.
"""
from typing import Dict, Any

from gufe import ChemicalSystem, Context,  ProtocolUnit, AtomMapping


class SetupUnit(ProtocolUnit):
    """
    Initial unit of the protocol. Creates a hybrid topology for the protein mutation and
    generates the OpenMM.
    """
    def _execute(ctx: Context, *, state_a: ChemicalSystem, state_b: ChemicalSystem, mapping: AtomMapping, **inputs) -> Dict[str, Any]:
        """

        Parameters
        ----------
        state_a
        state_b
        mapping
        inputs

        Returns
        -------

        """
        return NotImplementedError