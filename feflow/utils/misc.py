"""
Miscellaneous utility functions to extract data from gufe objects (and others)
"""

from typing import Type
import gufe
import numpy as np
import openmm.app


def register_ff_parameters_template(system_generator, charge_settings, openff_mols):
    """
    Register force field parameters in the system generator using provided charge settings
    and OpenFF molecules.

    This utility function assigns partial charges to the specified OpenFF molecules using
    the provided charge settings, then forces the creation of force field parameters by
    registering the templates with the system generator. This ensures the required force field
    templates are available prior to solvating the system.

    Parameters
    ----------
    system_generator : openmmforcefields.generators.SystemGenerator
        The system generator used to create force field parameters for the molecules.
    charge_settings : feflow.settings.ChargeSettings
        Settings for partial charge assignment, including the method, toolkit backend,
        number of conformers to generate, and optional NAGL model.
    openff_mols : list[openff.toolkit.Molecule]
        List of OpenFF molecules for which force field parameters are registered.

    Notes
    -----
    - Partial charges are assigned to the molecules using the OpenFF Toolkit based on the
      specified `charge_settings`.
    - Force field templates are registered by creating a system for each molecule with
      the system generator. This is necessary to ensure templates are available before
      solvating or otherwise processing the system.

    Examples
    --------
    >>> from openmmforcefields.generators import SystemGenerator
    >>> from openff.toolkit import Molecule
    >>> from feflow.settings import OpenFFPartialChargeSettings as ChargeSettings
    >>>
    >>> system_generator = SystemGenerator(small_molecule_forcefield="openff-2.1.0")
    >>> charge_settings = ChargeSettings(
    >>>     partial_charge_method="am1bcc",
    >>>     off_toolkit_backend="openeye",
    >>>     number_of_conformers=1,
    >>>     nagl_model=None
    >>> )
    >>> openff_mols = [Molecule.from_smiles("CCO"), Molecule.from_smiles("CCN")]
    >>> register_ff_parameters_template(system_generator, charge_settings, openff_mols)
    """
    from feflow.utils.charge import assign_offmol_partial_charges

    # Assign partial charges to all small mols -- we use openff for that
    for mol in openff_mols:
        assign_offmol_partial_charges(
            offmol=mol,
            overwrite=False,
            method=charge_settings.partial_charge_method,
            toolkit_backend=charge_settings.off_toolkit_backend,
            generate_n_conformers=charge_settings.number_of_conformers,
            nagl_model=charge_settings.nagl_model,
        )
        # Force the creation of parameters
        # This is necessary because we need to have the FF templates
        # registered ahead of solvating the system.
        system_generator.create_system(mol.to_topology().to_openmm(), molecules=[mol])


# TODO: To be revisited when gufe components have a topology consensus
def generate_omm_top_from_component(
    comp: gufe.SmallMoleculeComponent | gufe.ProteinComponent,
):
    """
    Generate an OpenMM `Topology` object from a given `SmallMoleculeComponent` or
    `ProteinComponent`.

    This function attempts to generate an OpenMM `Topology` object from the provided
    component. It handles both components that directly support conversion to an
    OpenMM topology (`to_openmm_topology`) and those that require an intermediate
    conversion through OpenFF (`to_openff().to_topology().to_openmm()`).

    Parameters
    ----------
    comp : gufe.SmallMoleculeComponent | gufe.ProteinComponent
        The component to be converted into an OpenMM `Topology`. Supported components include
        `SmallMoleculeComponent` and `ProteinComponent`.

    Returns
    -------
    openmm.app.Topology
        The corresponding OpenMM `Topology` object for the given component.

    Raises
    ------
    AttributeError
        If the component does not support the necessary conversion methods.
    """

    if isinstance(comp, gufe.ProteinComponent):
        topology = comp.to_openmm_topology()
    elif isinstance(comp, gufe.SmallMoleculeComponent):
        topology = comp.to_openff().to_topology().to_openmm()
    else:
        raise ValueError(
            f"Expected {gufe.SmallMoleculeComponent} or {gufe.ProteinComponent}. Received {type(comp)}."
        )

    return topology


# TODO: Maybe gufe components should have a way to get positions?
def get_positions_from_component(
    comp: gufe.SmallMoleculeComponent | gufe.ProteinComponent,
):
    """
    Retrieve the positions of atoms in a component as an OpenMM Quantity.

    This function tries to get the atomic positions from the component. If the component has
    a method `to_openmm_positions()`, it uses that to fetch the positions. If the component
    doesn't have that method (i.e., it doesn't support OpenMM directly), it falls back to
    extracting the positions from the OpenFF conformers and takes the first conformer.

    Parameters
    ----------
    comp : gufe.SmallMoleculeComponent | gufe.ProteinComponent
        The component (small molecule or protein) for which atomic positions are required.

    Returns
    -------
    openmm.Quantity
        A quantity representing the atomic positions in OpenMM format.

    Raises
    ------
    AttributeError
        If neither `to_openmm_positions()` nor OpenFF conformers are available.
    """
    # NOTE: Could potentially be done with rdkit if we want to rely solely on it, something like:
    # # Retrieve the first conformer (if multiple conformers exist)
    # mol = comp.to_rdkit()
    # conformer = mol.GetConformer(0)
    # conformer.GetPositions()
    from openff.units import ensure_quantity

    try:
        positions = comp.to_openmm_positions()
    except AttributeError:
        positions = comp.to_openff().conformers[0]

    return ensure_quantity(positions, "openmm")


def get_chain_residues_from_resids(
    topology: openmm.app.Topology, residue_indices: list[int]
):
    """
    Return all residues belonging to the same chains as a given set of residue indices.

    The function first identifies which chains the given residues belong to,
    then filters all residues in the topology to return only those from the
    same chains.

    Parameters
    ----------
    topology : openmm.app.Topology
        The topology containing the residues to inspect.
    residue_indices : sequence of int
        Indices of residues whose chains should be used to select other residues.

    Returns
    -------
    list of openmm.app.topology.Residue
        All residues from the topology that belong to any of the chains
        containing the specified residue indices. Order is preserved.
    """
    residues_list = list(topology.residues())  # All residues
    chains = {residues_list[i].chain for i in residue_indices}
    return np.array([res.index for res in residues_list if res.chain in chains])
