"""
Implementation of protein mutation protocol based on Nonequilibrium Cycling, using OpenMM as
MD engine.
"""
from pathlib import Path
from typing import Dict, Any

from gufe import ChemicalSystem, Context,  ProtocolUnit, AtomMapping


def mutate_with_pdbfixer(filename: str | Path, mutation_spec: str, chain_id: str = "A", output_pdb=None):
    """
    Takes a pdb path and applies mutation using pdbfixer.

    Parameters
    ----------
    filename
    mutation_spec
    chain_id
    output_pdb

    Returns
    -------

    """
    import pdbfixer
    from openmm.app import PDBFile
    fixer = pdbfixer.PDBFixer(filename=str(filename))
    # make sure your structure is complete
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    if bool(fixer.missingAtoms) or bool(fixer.missingResidues):
        raise ValueError(f"Your structure has the following missing elements. Missing residues: "
                         f"{fixer.missingResidues}; missing atoms: {fixer.missingAtoms}. Please fix.")
    fixer.applyMutations(mutations=[mutation_spec], chain_id=chain_id)
    omm_topology = fixer.topology
    omm_positions = fixer.positions

    # Write pdb file if specified
    if output_pdb:
        with open(output_pdb, "w") as out_file:
            PDBFile.writeFile(omm_topology, omm_positions, out_file)

    return omm_topology, omm_positions


class SetupUnit(ProtocolUnit):
    """
    Initial un      it of the protocol. Creates a hybrid topology for the protein mutation and
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