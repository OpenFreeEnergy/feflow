"""
Miscellaneous utility functions to extract data from gufe objects (and others)
"""

from typing import Type
import gufe


# TODO: should this be a method for the gufe.ChemicalSystem class?
def get_typed_components(system: gufe.ChemicalSystem, comptype: Type[gufe.Component]) -> set[
    gufe.Component]:
    """
    Retrieve all components of a specific type from a `gufe.ChemicalSystem`.

    This function searches the components within the provided chemical system
    and returns a list of all components matching the specified type.

    Parameters
    ----------
    system : gufe.ChemicalSystem
        The chemical system from which to extract components.
    comptype : Type[gufe.Component]
        The type of component to search for, such as `ProteinComponent`,
        `SmallMoleculeComponent`, or `SolventComponent`.

    Returns
    -------
    set[gufe.Component]
        A set of unique components matching the specified type. If no components
        of the given type are found, an empty set is returned.

    """
    if not issubclass(comptype, gufe.Component):
        raise TypeError(f"`comptype` must be a subclass of `gufe.Component`. Got: {comptype}")

    ret_comps = {comp for comp in system.values()
                 if isinstance(comp, comptype)}

    return ret_comps


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
        system_generator.create_system(
            mol.to_topology().to_openmm(), molecules=[mol]
        )
