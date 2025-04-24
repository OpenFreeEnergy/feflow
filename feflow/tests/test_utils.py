"""
Module to test utility functions in feflow.utils
"""

from gufe.components import SmallMoleculeComponent, ProteinComponent, SolventComponent
from feflow.utils.misc import get_typed_components, register_ff_parameters_template


def test_get_typed_components_vacuum(benzene_vacuum_system):
    """Test extracting typed components from a vacuum phase chemical system.
    One that only has a SmallMoleculeComponent.
    """
    small_mol_comps = get_typed_components(
        benzene_vacuum_system, SmallMoleculeComponent
    )
    protein_comps = get_typed_components(benzene_vacuum_system, ProteinComponent)
    solvent_comps = get_typed_components(benzene_vacuum_system, SolventComponent)

    assert (
        len(small_mol_comps) == 1
    ), f"Expected one (1) small molecule component in solvent system. Found {len(small_mol_comps)}"
    assert (
        len(protein_comps) == 0
    ), "Found protein component(s) in vacuum system. Expected none."
    assert (
        len(solvent_comps) == 0
    ), "Found solvent component(s) in vacuum system. Expected none."


def test_get_typed_components_solvent(benzene_solvent_system):
    """Test extracting typed components from a solvent phase chemical system.
    One that has a single SmallMoleculeComponent and a single SolventComponent.
    """
    small_mol_comps = get_typed_components(
        benzene_solvent_system, SmallMoleculeComponent
    )
    protein_comps = get_typed_components(benzene_solvent_system, ProteinComponent)
    solvent_comps = get_typed_components(benzene_solvent_system, SolventComponent)

    assert (
        len(small_mol_comps) == 1
    ), f"Expected one (1) small molecule component in vacuum system. Found {len(small_mol_comps)}."
    assert (
        len(protein_comps) == 0
    ), "Found protein component(s) in solvent system. Expected none."
    assert (
        len(solvent_comps) == 1
    ), f"Expected one (1) solvent component in solvent system. Found {len(solvent_comps)}."


def test_register_ff_parameters_template(
    toluene_solvent_system, short_settings, tmp_path
):
    from openff.toolkit import Molecule
    from openfe.protocols.openmm_utils import system_creation
    from openmmforcefields.generators import SystemGenerator
    from feflow.settings import OpenFFPartialChargeSettings as ChargeSettings
    from openfe.protocols.openmm_utils.system_validation import get_components

    solvent_comp, receptor_comp, small_mols_a = get_components(toluene_solvent_system)

    system_generator = system_creation.get_system_generator(
        forcefield_settings=short_settings.forcefield_settings,
        thermo_settings=short_settings.thermo_settings,
        integrator_settings=short_settings.integrator_settings,
        has_solvent=solvent_comp is not None,
        cache=tmp_path,
    )

    system_generator = SystemGenerator(small_molecule_forcefield="openff-2.1.0")
    charge_settings = ChargeSettings(
        partial_charge_method="am1bcc",
        off_toolkit_backend="ambertools",
        number_of_conformers=1,
        nagl_model=None,
    )
    openff_mols = [Molecule.from_smiles("CCO"), Molecule.from_smiles("CCN")]
    register_ff_parameters_template(system_generator, charge_settings, openff_mols)
