"""
Module to implement the basic unit testing for the hybrid topology implementation.
Specifically, regarding the HybridTopologyFactory object.
"""
import pytest

from feflow.utils.hybrid_topology import HybridTopologyFactory


@pytest.fixture(scope="module")
def standard_system_generator():
    """
    Fixture to create a standard/default system generator based on commonly used options for
    force fields, temperature and pressure. That is, amber forcefields for proteins, openff 2.1.0
    for small molecules, 1 bar pressure and 300k temperature.

    Returns
    -------
    generator: openmmforcefields.generators.SystemGenerator
        Dictionary with the key, value pairs to directly use when creating an instance of
        openmmforcefields.generators.SystemGenerator.
    """
    from openmm.app import NoCutoff, PME
    from openmm import unit as omm_unit
    from openmm import MonteCarloBarostat
    from openmmforcefields.generators import SystemGenerator

    sys_gen_config = {}
    sys_gen_config["forcefields"] = ["amber/ff14SB.xml",
                                     "amber/tip3p_standard.xml",
                                     "amber/tip3p_HFE_multivalent.xml",
                                     "amber/phosaa10.xml"]
    sys_gen_config["small_molecule_forcefield"] = "openff-2.1.0"
    sys_gen_config["nonperiodic_forcefield_kwargs"] = {
        "nonbondedMethod": NoCutoff,
    }
    sys_gen_config["periodic_forcefield_kwargs"] = {
        "nonbondedMethod": PME,
        "nonbondedCutoff": 1.0 * omm_unit.nanometer,
    }
    sys_gen_config["barostat"] = MonteCarloBarostat(1 * omm_unit.bar, 300 * omm_unit.kelvin)

    generator = SystemGenerator(**sys_gen_config)

    return generator


def test_hybrid_topology_benzene_phenol(benzene, toluene, mapping_benzene_toluene,
                                        standard_system_generator):
    """
    Test the creation of a HybridTopologyFactory object from scratch from a benzene to toluene
    transformation.

    Tests that we can create a HTF from scratch and checks that the difference in the number of
    atoms in the Hybrid topology initial and final states is the expected one.

    Returns
    -------
    None
    """
    benzene_offmol = benzene.to_openff()
    toluene_offmol = toluene.to_openff()
    # Create Openmm topologies from openff topologies
    off_top_benzene = benzene_offmol.to_topology()
    off_top_phenol = toluene_offmol.to_topology()
    # Create openmm topologies initial and final states
    initial_top = off_top_benzene.to_openmm()
    final_top = off_top_phenol.to_openmm()
    # Create openmm systems with the small molecules
    system_generator = standard_system_generator
    initial_system = system_generator.create_system(initial_top, molecules=[benzene_offmol])
    final_system = system_generator.create_system(final_top, molecules=[toluene_offmol])
    initial_positions = benzene_offmol.conformers[-1].to_openmm()
    final_positions = toluene_offmol.conformers[-1].to_openmm()
    # mapping
    mapping = mapping_benzene_toluene.componentA_to_componentB  # Initial to final map
    initial_to_final_atom_map = mapping
    initial_to_final_core_atom_map = mapping
    # Instantiate HTF
    htf = HybridTopologyFactory(
        initial_system,
        initial_positions,
        initial_top,
        final_system,
        final_positions,
        final_top,
        initial_to_final_atom_map,
        initial_to_final_core_atom_map
    )

    # Validate number of atoms in hybrid topology end systems
    n_atoms_diff = benzene_offmol.n_atoms - toluene_offmol.n_atoms  # Initial - Final -- Sign/order matters
    initial_htf_n_atoms = len(htf.initial_atom_indices)
    final_htf_n_atoms = len(htf.final_atom_indices)
    assert initial_htf_n_atoms - final_htf_n_atoms == n_atoms_diff, \
        "Different number of atoms in HTF compared to original molecules."

    # Validate common atoms include 6 carbon atoms

