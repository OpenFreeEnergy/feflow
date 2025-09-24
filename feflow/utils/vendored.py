"""
Vendored from: https://github.com/OpenFreeEnergy/openfe/blob/main/openfe/protocols/openmm_utils/system_creation.py
Original version: v1.2.0 (commit 48dcbb26)
Date vendored: 2025-01-23
License: MIT

Modifications made:
- Allowing multiple optional components (protein, smallmols or solvent)

Original copyright notice:
Copyright (c) 2025 Open Free Energy
"""

import itertools
import warnings
from copy import deepcopy
from typing import Optional
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from gufe import Component, ProteinComponent, SmallMoleculeComponent, SolventComponent
from gufe.settings import OpenMMSystemGeneratorFFSettings
from openff.units.openmm import to_openmm, ensure_quantity
from openfe.protocols.openmm_utils.system_creation import ModellerReturn
from openfe.protocols.openmm_utils.omm_settings import OpenMMSolvationSettings
from openmm.app import Element, ForceField, Modeller, Topology


def get_omm_modeller(
    protein_comps: Optional[Iterable[ProteinComponent] | ProteinComponent],
    solvent_comps: Optional[Iterable[SolventComponent] | SolventComponent],
    small_mols: Optional[Iterable[SmallMoleculeComponent] | SmallMoleculeComponent],
    omm_forcefield: ForceField,
    solvent_settings: OpenMMSolvationSettings,
) -> ModellerReturn:
    """
    Generate an OpenMM Modeller class based on a potential input ProteinComponent,
    SolventComponent, and a set of small molecules.

    Parameters
    ----------
    protein_comps : Optional[Iterable[ProteinComponent] or ProteinComponent]
      Protein Component, if it exists.
    solvent_comps : Optional[Iterable[SolventComponent] or SolventComponent]
      Solvent Component, if it exists.
    small_mols : Optional[Iterable[SmallMoleculeComponent] or SmallMoleculeComponent]
      Small molecules to add.
    omm_forcefield : openmm.app.ForceField
      ForceField object for system.
    solvent_settings : SolvationSettings
      Solvation settings.

    Returns
    -------
    system_modeller : app.Modeller
      OpenMM Modeller object generated from ProteinComponent and
      OpenFF Molecules.
    component_resids : dict[Component, npt.NDArray]
      Dictionary of residue indices for each component in system.
    """
    component_resids = {}

    def _add_small_mol(
        comp, mol, system_modeller: Modeller, comp_resids: dict[Component, npt.NDArray]
    ):
        """
        Helper method to add OFFMol to an existing Modeller object and
        update a dictionary tracking residue indices for each component.
        """
        omm_top = mol.to_topology().to_openmm()
        system_modeller.add(omm_top, ensure_quantity(mol.conformers[0], "openmm"))

        nres = omm_top.getNumResidues()
        resids = [res.index for res in system_modeller.topology.residues()]
        comp_resids[comp] = np.array(resids[-nres:])

    # Create empty modeller
    system_modeller = Modeller(Topology(), [])

    # We first add all the protein components, if any
    if protein_comps:
        try:
            protein_comps = iter(protein_comps)
        except TypeError:
            protein_comps = {protein_comps}  # make it a set/iterable with the comp
        for protein_comp in protein_comps:
            system_modeller.add(
                protein_comp.to_openmm_topology(), protein_comp.to_openmm_positions()
            )
            # add missing virtual particles (from crystal waters)
            system_modeller.addExtraParticles(omm_forcefield)
            component_resids[protein_comp] = np.array(
                [r.index for r in system_modeller.topology.residues()]
            )
            # if we solvate temporarily rename water molecules to 'WAT'
            # see openmm issue #4103
            if solvent_comps is not None:
                for r in system_modeller.topology.residues():
                    if r.name == "HOH":
                        r.name = "WAT"

    # Now loop through small mols
    if small_mols:
        try:
            small_mols = iter(small_mols)
        except TypeError:
            small_mols = {small_mols}  # make it a set/iterable with the comp
        for small_mol_comp in small_mols:
            _add_small_mol(
                small_mol_comp,
                small_mol_comp.to_openff(),
                system_modeller,
                component_resids,
            )

    # Add solvent if neeeded
    if solvent_comps:
        # Making it a list to make our life easier -- TODO: Maybe there's a better type for this
        try:
            solvent_comps = list(set(solvent_comps))  # if given iterable
        except TypeError:
            solvent_comps = [solvent_comps]  # if not iterable, given single obj
        # TODO: Support multiple solvent components? Is there a use case for it?
        # Error out when we iter(have more than one solvent component in the states/systems
        if len(solvent_comps) > 1:
            raise ValueError(
                "More than one solvent component found in systems. Only one supported."
            )
        solvent_comp = solvent_comps[0]  # Get the first (and only?) solvent component
        # Do unit conversions if necessary
        solvent_padding = None
        box_size = None
        box_vectors = None

        if solvent_settings.solvent_padding is not None:
            solvent_padding = to_openmm(solvent_settings.solvent_padding)

        if solvent_settings.box_size is not None:
            box_size = to_openmm(solvent_settings.box_size)

        if solvent_settings.box_vectors is not None:
            box_vectors = to_openmm(solvent_settings.box_vectors)

        system_modeller.addSolvent(
            omm_forcefield,
            model=solvent_settings.solvent_model,
            padding=solvent_padding,
            positiveIon=solvent_comp.positive_ion,
            negativeIon=solvent_comp.negative_ion,
            ionicStrength=to_openmm(solvent_comp.ion_concentration),
            neutralize=solvent_comp.neutralize,
            boxSize=box_size,
            boxVectors=box_vectors,
            boxShape=solvent_settings.box_shape,
            numAdded=solvent_settings.number_of_solvent_molecules,
        )

        all_resids = np.array([r.index for r in system_modeller.topology.residues()])

        existing_resids = np.concatenate(
            [resids for resids in component_resids.values()]
        )

        component_resids[solvent_comp] = np.setdiff1d(all_resids, existing_resids)
        # undo rename of pre-existing waters
        for r in system_modeller.topology.residues():
            if r.name == "WAT":
                r.name = "HOH"

    return system_modeller, component_resids


# Vendored from OpenFreeEnergy/openfe at 2025-09-24
def _get_indices(topology, resids):
    """
    Get the atoms indices from an array of residue indices in an OpenMM Topology

    Parameters
    ----------
    topology : openmm.app.Topology
        Topology to search from.
    residue_name : str
        Name of the residue to get the indices for.
    """
    # TODO: remove, this shouldn't be necessary anymore
    if len(resids) > 1:
        raise ValueError("multiple residues were found")

    # create list of openmm residues
    top_res = [r for r in topology.residues() if r.index in resids]

    # get a list of all atoms in residues
    top_atoms = list(itertools.chain.from_iterable(r.atoms() for r in top_res))

    return [at.index for at in top_atoms]


# Vendored from OpenFreeEnergy/openfe at 2025-09-24
def _remove_constraints(
    old_to_new_atom_map, old_system, old_topology, new_system, new_topology
):
    """
    Adapted from Perses' Topology Proposal. Adjusts atom mapping to account for
    any bonds that are constrained but change in length.

    Parameters
    ----------
    old_to_new_atom_map : dict of int : int
        Atom mapping between the old and new systems.
    old_system : openmm.app.System
        System of the "old" alchemical state.
    old_topology : openmm.app.Topology
        Topology of the "old" alchemical state.
    new_system : openmm.app.System
        System of the "new" alchemical state.
    new_topology : openmm.app.Topology
        Topology of the "new" alchemical state.

    Returns
    -------
    no_const_old_to_new_atom_map : dict of int : int
        Adjusted version of the input mapping but with atoms involving changes
        in lengths of constrained bonds removed.

    TODO
    ----
    * Very slow, needs refactoring
    * Can we drop having topologies as inputs here?
    """
    no_const_old_to_new_atom_map = deepcopy(old_to_new_atom_map)

    h_elem = Element.getByAtomicNumber(1)
    old_H_atoms = {
        i
        for i, atom in enumerate(old_topology.atoms())
        if atom.element == h_elem and i in old_to_new_atom_map
    }
    new_H_atoms = {
        i
        for i, atom in enumerate(new_topology.atoms())
        if atom.element == h_elem and i in old_to_new_atom_map.values()
    }

    def pick_H(i, j, x, y) -> int:
        """Identify which atom to remove to resolve constraint violation

        i maps to x, j maps to y

        Returns either i or j (whichever is H) to remove from mapping
        """
        if i in old_H_atoms or x in new_H_atoms:
            return i
        elif j in old_H_atoms or y in new_H_atoms:
            return j
        else:
            raise ValueError(
                f"Couldn't resolve constraint demapping for atoms"
                f" A: {i}-{j} B: {x}-{y}"
            )

    old_constraints: dict[[int, int], float] = {}
    for idx in range(old_system.getNumConstraints()):
        atom1, atom2, length = old_system.getConstraintParameters(idx)

        if atom1 in old_to_new_atom_map and atom2 in old_to_new_atom_map:
            old_constraints[atom1, atom2] = length

    new_constraints = {}
    for idx in range(new_system.getNumConstraints()):
        atom1, atom2, length = new_system.getConstraintParameters(idx)

        if (
            atom1 in old_to_new_atom_map.values()
            and atom2 in old_to_new_atom_map.values()
        ):
            new_constraints[atom1, atom2] = length

    # there are two reasons constraints would invalidate a mapping entry
    # 1) length of constraint changed (but both constrained)
    # 2) constraint removed to harmonic bond (only one constrained)
    to_del = []
    for (i, j), l_old in old_constraints.items():
        x, y = old_to_new_atom_map[i], old_to_new_atom_map[j]

        try:
            l_new = new_constraints.pop((x, y))
        except KeyError:
            try:
                l_new = new_constraints.pop((y, x))
            except KeyError:
                # type 2) constraint doesn't exist in new system
                to_del.append(pick_H(i, j, x, y))
                continue

        # type 1) constraint length changed
        if l_old != l_new:
            to_del.append(pick_H(i, j, x, y))

    # iterate over new_constraints (we were .popping items out)
    # (if any left these are type 2))
    if new_constraints:
        new_to_old = {v: k for k, v in old_to_new_atom_map.items()}

        for x, y in new_constraints:
            i, j = new_to_old[x], new_to_old[y]

            to_del.append(pick_H(i, j, x, y))

    for idx in to_del:
        del no_const_old_to_new_atom_map[idx]

    return no_const_old_to_new_atom_map


# Vendored from OpenFreeEnergy/openfe at 2025-09-24
def get_system_mappings(
    old_to_new_atom_map,
    old_system,
    old_topology,
    old_resids,
    new_system,
    new_topology,
    new_resids,
    fix_constraints=True,
):
    """
    From a starting alchemical map between two molecules, get the mappings
    between two alchemical end state systems.

    Optionally, also fixes the mapping to account for a) element changes, and
    b) changes in bond lengths for constraints.

    Parameters
    ----------
    old_to_new_atom_map : dict of int : int
        Atom mapping between the old and new systems.
    old_system : openmm.app.System
        System of the "old" alchemical state.
    old_topology : openmm.app.Topology
        Topology of the "old" alchemical state.
    old_resids : npt.NDArray
        Residue ids of the alchemical residues in the "old" topology.
    new_system : openmm.app.System
        System of the "new" alchemical state.
    new_topology : openmm.app.Topology
        Topology of the "new" alchemical state.
    new_resids : npt.NDArray
        Residue ids of the alchemical residues in the "new" topology.
    fix_constraints : bool, default True
        Whether to fix the atom mapping by removing any atoms which are
        involved in constrained bonds that change length across the alchemical
        change.

    Returns
    -------
    mappings : dictionary
        A dictionary with all the necessary mappings for the two systems.
        These include:
            1. old_to_new_atom_map
              This includes all the atoms mapped between the two systems
              (including non-core atoms, i.e. environment).
            2. new_to_old_atom_map
              The inverted dictionary of old_to_new_atom_map
            3. old_to_new_core_atom_map
              The atom mapping of the "core" atoms (i.e. atoms in alchemical
              residues) between the old and new systems
            4. new_to_old_core_atom_map
              The inverted dictionary of old_to_new_core_atom_map
            5. old_to_new_env_atom_map
              The atom mapping of solely the "environment" atoms between the
              old and new systems.
            6. new_to_old_env_atom_map
              The inverted dictionaryu of old_to_new_env_atom_map.
            7. old_mol_indices
              Indices of the alchemical molecule in the old system.
              Note: This will not contain the indices of any alchemical waters!
            8. new_mol_indices
              Indices of the alchemical molecule in the new system.
              Note: This will not contain the indices of any alchemical waters!
    """
    # Get the indices of the atoms in the alchemical residue of interest for
    # both the old and new systems
    old_at_indices = _get_indices(old_topology, old_resids)
    new_at_indices = _get_indices(new_topology, new_resids)

    # We assume that the atom indices are linear in the residue so we shift
    # by the index of the first atom in each residue
    adjusted_old_to_new_map = {}
    for key, value in old_to_new_atom_map.items():
        shift_old = old_at_indices[0] + key
        shift_new = new_at_indices[0] + value
        adjusted_old_to_new_map[shift_old] = shift_new

    # TODO: the original intent here was to apply over the full mapping of all
    # the atoms in the two systems. For now we are only doing the alchemical
    # residues. We might want to change this as necessary in the future.
    if not fix_constraints:
        wmsg = (
            "Not attempting to fix atom mapping to account for "
            "constraints. Please note that core atoms which have "
            "constrained bonds and changing bond lengths are not allowed."
        )
        warnings.warn(wmsg)
    else:
        adjusted_old_to_new_map = _remove_constraints(
            adjusted_old_to_new_map, old_system, old_topology, new_system, new_topology
        )

    # We return a dictionary with all the necessary mappings (as they are
    # needed downstream). These include:
    #  1. old_to_new_atom_map
    #     This includes all the atoms mapped between the two systems
    #     (including non-core atoms, i.e. environment).
    #  2. new_to_old_atom_map
    #     The inverted dictionary of old_to_new_atom_map
    #  3. old_to_new_core_atom_map
    #     The atom mapping of the "core" atoms (i.e. atoms in alchemical
    #     residues) between the old and new systems
    #  4. new_to_old_core_atom_map
    #     The inverted dictionary of old_to_new_core_atom_map
    #  5. old_to_new_env_atom_map
    #     The atom mapping of solely the "environment" atoms between the old
    #     and new systems.
    #  6. new_to_old_env_atom_map
    #     The inverted dictionaryu of old_to_new_env_atom_map.

    # Because of how we append the topologies, we can assume that the last
    # residue in the "new" topology is the ligand, just to be sure we check
    # this here - temp fix for now
    for at in new_topology.atoms():
        if at.index > new_at_indices[-1]:
            raise ValueError("residues are appended after the new ligand")

    # We assume that all the atoms up until the first ligand atom match
    # except from the indices of the ligand in the old topology.
    new_to_old_all_map = {}
    old_mol_offset = len(old_at_indices)
    for i in range(new_at_indices[0]):
        if i >= old_at_indices[0]:
            old_idx = i + old_mol_offset
        else:
            old_idx = i
        new_to_old_all_map[i] = old_idx

    # At this point we only have environment atoms so make a copy
    new_to_old_env_map = deepcopy(new_to_old_all_map)

    # Next we append the contents of the "core" map we already have
    for key, value in adjusted_old_to_new_map.items():
        # reverse order because we are going new->old instead of old->new
        new_to_old_all_map[value] = key

    # Now let's create our output dictionary
    mappings = {}
    mappings["new_to_old_atom_map"] = new_to_old_all_map
    mappings["old_to_new_atom_map"] = {v: k for k, v in new_to_old_all_map.items()}
    mappings["new_to_old_core_atom_map"] = {
        v: k for k, v in adjusted_old_to_new_map.items()
    }
    mappings["old_to_new_core_atom_map"] = adjusted_old_to_new_map
    mappings["new_to_old_env_atom_map"] = new_to_old_env_map
    mappings["old_to_new_env_atom_map"] = {v: k for k, v in new_to_old_env_map.items()}
    mappings["old_mol_indices"] = old_at_indices
    mappings["new_mol_indices"] = new_at_indices

    return mappings
