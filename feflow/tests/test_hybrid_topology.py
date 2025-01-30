"""
Module to implement the basic unit testing for the hybrid topology implementation.
Specifically, regarding the HybridTopologyFactory object.
More oriented to testing code functionality than science correctness.
"""

import pytest

from feflow.utils.hybrid_topology import HybridTopologyFactory
from feflow.tests.utils import extract_htf_data

import mdtraj as mdt
import numpy as np
from openmm import unit as omm_unit
from openmm.app import NoCutoff, PME
from openmm import MonteCarloBarostat, NonbondedForce, CustomNonbondedForce
from openmmforcefields.generators import SystemGenerator
from openff.units.openmm import to_openmm, from_openmm, ensure_quantity
from perses.tests import utils as perses_utils


@pytest.fixture(scope="module")
def standard_system_generator():
    """
    Fixture to create a standard/default system generator based on commonly used options for
    force fields, temperature and pressure. That is, amber forcefields for proteins, openff 2.1.0
    for small molecules, 1 bar pressure and 300k temperature.

    Returns
    -------
    generator: openmmforcefields.generators.SystemGenerator
        SystemGenerator object.
    """
    sys_gen_config = {}
    sys_gen_config["forcefields"] = [
        "amber/ff14SB.xml",
        "amber/tip3p_standard.xml",
        "amber/tip3p_HFE_multivalent.xml",
        "amber/phosaa10.xml",
    ]
    sys_gen_config["small_molecule_forcefield"] = "openff-2.1.0"
    sys_gen_config["nonperiodic_forcefield_kwargs"] = {
        "nonbondedMethod": NoCutoff,
    }
    sys_gen_config["periodic_forcefield_kwargs"] = {
        "nonbondedMethod": PME,
        "nonbondedCutoff": 1.0 * omm_unit.nanometer,
    }
    sys_gen_config["barostat"] = MonteCarloBarostat(
        1 * omm_unit.bar, 300 * omm_unit.kelvin
    )

    generator = SystemGenerator(**sys_gen_config)

    return generator


class TestHybridTopologyFactory:
    """Class to test the base/vanilla HybridTopologyFactory object"""

    def test_custom_nonbonded_cutoff(self):
        """
        Test that nonbonded cutoff gets propagated to the custom nonbonded forces generated in the HTF via the
        _add_nonbonded_force_terms method.

        Creates an HTF and manually changes the cutoff in the OLD system of the hybrid topology factory and checks the
        expected behavior with both running or not running the referenced method.
        """
        from openmm import NonbondedForce, CustomNonbondedForce

        # TODO: we should probably make a fixture with the following top proposal and factory
        (
            topology_proposal,
            current_positions,
            new_positions,
        ) = perses_utils.generate_solvated_hybrid_test_topology(
            current_mol_name="propane", proposed_mol_name="pentane", vacuum=False
        )
        # Extract htf data from proposal
        htf_data = extract_htf_data(topology_proposal)
        hybrid_factory = HybridTopologyFactory(
            old_positions=current_positions,
            new_positions=new_positions,
            **htf_data,
            use_dispersion_correction=True,
        )
        old_system_forces = hybrid_factory._old_system_forces
        hybrid_system_forces = hybrid_factory.hybrid_system.getForces()
        old_nonbonded_forces = [
            force for force in old_system_forces if isinstance(force, NonbondedForce)
        ]
        hybrid_custom_nonbonded_forces = [
            force
            for force in hybrid_system_forces
            if isinstance(force, CustomNonbondedForce)
        ]
        # Modify the cutoff for nonbonded forces in the OLD system (!)
        for force in old_nonbonded_forces:
            force.setCutoffDistance(force.getCutoffDistance() + 1 * omm_unit.nanometer)
            # Assert that the nb cutoff distance is different compared to the custom nb forces
            for custom_force in hybrid_custom_nonbonded_forces:
                assert (
                    custom_force.getCutoffDistance() != force.getCutoffDistance()
                ), "Expected different cutoff distances between NB and custom NB forces."
        # propagate the cutoffs
        hybrid_factory._add_nonbonded_force_terms()
        # Check now that cutoff match for all nonbonded forces (including custom)
        for force in old_nonbonded_forces:
            for custom_force in hybrid_custom_nonbonded_forces:
                assert (
                    custom_force.getCutoffDistance() == force.getCutoffDistance()
                ), "Expected equal cutoff distances between NB and custom NB forces."

    def test_hybrid_topology_benzene_phenol(
        self, benzene, toluene, mapping_benzene_toluene, standard_system_generator
    ):
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
        initial_system = system_generator.create_system(
            initial_top, molecules=[benzene_offmol]
        )
        final_system = system_generator.create_system(
            final_top, molecules=[toluene_offmol]
        )
        initial_positions = benzene_offmol.conformers[-1].to_openmm()
        final_positions = toluene_offmol.conformers[-1].to_openmm()
        # mapping
        mapping = (
            mapping_benzene_toluene.componentA_to_componentB
        )  # Initial to final map
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
            initial_to_final_core_atom_map,
        )

        # Validate number of atoms in hybrid topology end systems
        n_atoms_diff = (
            benzene_offmol.n_atoms - toluene_offmol.n_atoms
        )  # Initial - Final -- Sign/order matters
        initial_htf_n_atoms = len(htf.initial_atom_indices)
        final_htf_n_atoms = len(htf.final_atom_indices)
        assert (
            initial_htf_n_atoms - final_htf_n_atoms == n_atoms_diff
        ), "Different number of atoms in HTF compared to original molecules."

        # 16 atoms:
        # 11 common atoms, 1 extra hydrogen in benzene, 4 extra in toluene
        # 12 bonds in benzene + 4 extra toluene bonds
        assert len(list(htf.hybrid_topology.atoms)) == 16
        assert len(list(htf.hybrid_topology.bonds)) == 16
        # check that the omm_hybrid_topology has the right things
        assert len(list(htf.omm_hybrid_topology.atoms())) == 16
        assert len(list(htf.omm_hybrid_topology.bonds())) == 16
        # check that we can convert back the mdtraj hybrid_topology attribute
        ret_top = mdt.Topology.to_openmm(htf.hybrid_topology)
        assert len(list(ret_top.atoms())) == 16
        assert len(list(ret_top.bonds())) == 16

        # TODO: Validate common atoms include 6 carbon atoms


class TestHTFVirtualSites:
    @pytest.fixture(scope="module")
    def tip4p_system_generator(self):
        """
        SystemGenerator object with tip4p-ew water

        Returns
        -------
        generator: openmmforcefields.generators.SystemGenerator
            SystemGenerator object.
        """
        sys_gen_config = {}
        sys_gen_config["forcefields"] = [
            "amber/ff14SB.xml",
            "amber/tip4pew_standard.xml",
            "amber/phosaa10.xml",
        ]
        sys_gen_config["small_molecule_forcefield"] = "openff-2.1.0"
        sys_gen_config["nonperiodic_forcefield_kwargs"] = {
            "nonbondedMethod": NoCutoff,
        }
        sys_gen_config["periodic_forcefield_kwargs"] = {
            "nonbondedMethod": PME,
            "nonbondedCutoff": 1.0 * omm_unit.nanometer,
        }
        sys_gen_config["barostat"] = MonteCarloBarostat(
            1 * omm_unit.bar, 300 * omm_unit.kelvin
        )

        generator = SystemGenerator(**sys_gen_config)

        return generator

    @pytest.fixture(scope="module")
    def tip4p_benzene_to_toluene_htf(
        self, tip4p_system_generator, benzene, toluene, mapping_benzene_toluene
    ):
        """
        TODO: turn part of this into a method for creating HTFs?
        """
        from gufe import SolventComponent

        # TODO: change imports once utils get moved
        from openfe.protocols.openmm_utils import system_creation
        from openfe.protocols.openmm_rfe._rfe_utils import topologyhelpers
        from openfe.protocols.openmm_utils.omm_settings import OpenMMSolvationSettings

        benz_off = benzene.to_openff()
        tol_off = toluene.to_openff()

        solv_settings = OpenMMSolvationSettings()
        solv_settings.solvent_model = "tip4pew"

        for mol in [benz_off, tol_off]:
            tip4p_system_generator.create_system(
                mol.to_topology().to_openmm(), molecules=[mol]
            )

        # Create state A model & get relevant OpenMM objects
        benz_model, comp_resids = system_creation.get_omm_modeller(
            protein_comps=None,
            solvent_comp=SolventComponent(),
            small_mols={benzene: benz_off},
            omm_forcefield=tip4p_system_generator.forcefield,
            solvent_settings=solv_settings,
        )

        benz_topology = benz_model.getTopology()
        benz_positions = to_openmm(from_openmm(benz_model.getPositions()))
        benz_system = tip4p_system_generator.create_system(
            benz_topology, molecules=[benz_off]
        )

        # Now for state B
        tol_topology, tol_alchem_resids = topologyhelpers.combined_topology(
            benz_topology,
            tol_off.to_topology().to_openmm(),
            exclude_resids=comp_resids[benzene],
        )

        tol_system = tip4p_system_generator.create_system(
            tol_topology, molecules=[tol_off]
        )

        ligand_mappings = topologyhelpers.get_system_mappings(
            mapping_benzene_toluene.componentA_to_componentB,
            benz_system,
            benz_topology,
            comp_resids[benzene],
            tol_system,
            tol_topology,
            tol_alchem_resids,
        )

        tol_positions = topologyhelpers.set_and_check_new_positions(
            ligand_mappings,
            benz_topology,
            tol_topology,
            old_positions=benz_positions,
            insert_positions=to_openmm(tol_off.conformers[0]),
        )

        # Finally get the HTF
        hybrid_factory = HybridTopologyFactory(
            benz_system,
            benz_positions,
            benz_topology,
            tol_system,
            tol_positions,
            tol_topology,
            old_to_new_atom_map=ligand_mappings["old_to_new_atom_map"],
            old_to_new_core_atom_map=ligand_mappings["old_to_new_core_atom_map"],
        )

        return hybrid_factory

    def test_tip4p_particle_count(self, tip4p_benzene_to_toluene_htf):
        """
        Check that the particle count is conserved, i.e. no vsites are lost
        or double counted.
        """
        htf = tip4p_benzene_to_toluene_htf
        old_count = htf._old_system.getNumParticles()
        unique_new_count = len(htf._unique_new_atoms)
        hybrid_particle_count = htf.hybrid_system.getNumParticles()

        assert old_count + unique_new_count == hybrid_particle_count

    def test_tip4p_num_waters(self, tip4p_benzene_to_toluene_htf):
        """
        Check that the nuumber of virtual sites is equal to the number of
        waters
        """
        htf = tip4p_benzene_to_toluene_htf

        num_waters = len([r for r in htf._old_topology.residues() if r.name == "HOH"])

        virtual_sites = [
            ix
            for ix in range(htf.hybrid_system.getNumParticles())
            if htf.hybrid_system.isVirtualSite(ix)
        ]

        assert num_waters == len(virtual_sites)

    def test_tip4p_check_vsite_parameters(self, tip4p_benzene_to_toluene_htf):
        htf = tip4p_benzene_to_toluene_htf

        virtual_sites = [
            ix
            for ix in range(htf.hybrid_system.getNumParticles())
            if htf.hybrid_system.isVirtualSite(ix)
        ]

        # get the standard and custom nonbonded forces - one of each
        nonbond = [
            f for f in htf.hybrid_system.getForces() if isinstance(f, NonbondedForce)
        ][0]

        cust_nonbond = [
            f
            for f in htf.hybrid_system.getForces()
            if isinstance(f, CustomNonbondedForce)
        ][0]

        # loop through every virtual site and check that they have the
        # expected tip4p parameters
        for entry in virtual_sites:
            vs = htf.hybrid_system.getVirtualSite(entry)
            vs_mass = htf.hybrid_system.getParticleMass(entry)
            assert ensure_quantity(vs_mass, "openff").m == pytest.approx(0)
            vs_weights = [vs.getWeight(ix) for ix in range(vs.getNumParticles())]
            np.testing.assert_allclose(
                vs_weights, [0.786646558, 0.106676721, 0.106676721]
            )
            c, s, e = nonbond.getParticleParameters(entry)
            assert ensure_quantity(c, "openff").m == pytest.approx(-1.04844)
            assert ensure_quantity(s, "openff").m == 1
            assert ensure_quantity(e, "openff").m == 0

            s1, e1, s2, e2, i, j = cust_nonbond.getParticleParameters(entry)

            assert i == j == 0
            assert s1 == s2 == 1
            assert e1 == e2 == 0
