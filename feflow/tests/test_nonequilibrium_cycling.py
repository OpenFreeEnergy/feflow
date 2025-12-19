import pickle
from importlib.resources import files
from pathlib import Path

import pymbar.utils
import pytest

from feflow.protocols import NonEquilibriumCyclingProtocol
from feflow.settings import NonEquilibriumCyclingSettings
from gufe.protocols.protocoldag import ProtocolDAGResult, execute_DAG
from gufe.protocols.protocolunit import ProtocolUnitResult
from gufe.tokenization import JSON_HANDLER

import json

from feflow.tests.conftest import solvent_comp


def partial_charges_config():
    partial_charges_testing_matrix = {
        "am1bcc": ["ambertools", "openeye"],
        "am1bccelf10": ["openeye"],
        # "nagl": ["ambertools", "openeye", "rdkit"],  # TODO: Add NAGL when there are production models
        "espaloma": ["ambertools", "rdkit"],
    }
    # Navigate dictionary and yield method, backend pair
    for key, value in partial_charges_testing_matrix.items():
        for val in value:
            yield key, val


def _check_htf_charges(hybrid_topology_factory, charges_state_a, charges_state_b):
    """
    Utility function that verifies particle charges and their offsets in the hybrid system.

    This function ensures that the particle charges and their parameter offsets in the hybrid
    system match the expected values from benzene and toluene random charges.

    The logic is as follows:
        HTF creates two sets of nonbonded forces, a standard one (for the
        PME) and a custom one (for sterics).
        Here we specifically check charges, so we only concentrate on the
        standard NonbondedForce.
        The way the NonbondedForce is constructed is as follows:
        - unique old atoms:
         * The particle charge is set to the input molA particle charge
         * The chargeScale offset is set to the negative value of the molA
           particle charge (such that by scaling you effectively zero out
           the charge.
        - unique new atoms:
         * The particle charge is set to zero (doesn't exist in the starting
           end state).
         * The chargeScale offset is set to the value of the molB particle
           charge (such that by scaling you effectively go from 0 to molB
           charge).
        - core atoms:
         * The particle charge is set to the input molA particle charge
           (i.e. we start from a system that has molA charges).
         * The particle charge offset is set to the difference between
           the molB particle charge and the molA particle charge (i.e.
           we scale by that difference to get to the value of the molB
           particle charge).
    """
    import numpy as np
    from openmm import NonbondedForce
    from openff.units import unit, ensure_quantity

    # get the standard nonbonded force
    htf = hybrid_topology_factory
    hybrid_system = htf.hybrid_system
    nonbond = [f for f in hybrid_system.getForces() if isinstance(f, NonbondedForce)]
    assert len(nonbond) == 1

    # get the particle parameter offsets
    c_offsets = {}
    for i in range(nonbond[0].getNumParticleParameterOffsets()):
        offset = nonbond[0].getParticleParameterOffset(i)
        c_offsets[offset[1]] = ensure_quantity(offset[2], "openff")

    for i in range(hybrid_system.getNumParticles()):
        c, s, e = nonbond[0].getParticleParameters(i)
        # get the particle charge (c)
        c = ensure_quantity(c, "openff")
        # particle charge (c) is equal to molA particle charge
        # offset (c_offsets) is equal to -(molA particle charge)
        if i in htf._atom_classes["unique_old_atoms"]:
            idx = htf._hybrid_to_old_map[i]
            np.testing.assert_allclose(c, charges_state_a[idx])
            np.testing.assert_allclose(c_offsets[i], -charges_state_a[idx])
        # particle charge (c) is equal to 0
        # offset (c_offsets) is equal to molB particle charge
        elif i in htf._atom_classes["unique_new_atoms"]:
            idx = htf._hybrid_to_new_map[i]
            np.testing.assert_allclose(c, 0 * unit.elementary_charge)
            np.testing.assert_allclose(c_offsets[i], charges_state_b[idx])
        # particle charge (c) is equal to molA particle charge
        # offset (c_offsets) is equal to difference between molB and molA
        elif i in htf._atom_classes["core_atoms"]:
            old_i = htf._hybrid_to_old_map[i]
            new_i = htf._hybrid_to_new_map[i]
            c_exp = charges_state_b[new_i] - charges_state_a[old_i]
            np.testing.assert_allclose(c, charges_state_a[old_i])
            np.testing.assert_allclose(c_offsets[i], c_exp)


class TestNonEquilibriumCycling:
    @pytest.fixture
    def protocol_short(self, short_settings):
        return NonEquilibriumCyclingProtocol(settings=short_settings)

    @pytest.fixture
    def protocol_short_multiple_cycles(self, short_settings_multiple_cycles):
        return NonEquilibriumCyclingProtocol(settings=short_settings_multiple_cycles)

    @pytest.fixture
    def protocol_short_multiple_cycles_gpu(self, short_settings_multiple_cycles_gpu):
        return NonEquilibriumCyclingProtocol(
            settings=short_settings_multiple_cycles_gpu
        )

    @pytest.fixture
    def protocol_dag_result(
        self,
        protocol_short,
        benzene_vacuum_system,
        toluene_vacuum_system,
        mapping_benzene_toluene,
        tmpdir,
    ):
        dag = protocol_short.create(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            name="Short vacuum transformation",
            mapping=mapping_benzene_toluene,
        )

        with tmpdir.as_cwd():
            shared = Path("shared")
            shared.mkdir()

            scratch = Path("scratch")
            scratch.mkdir()

            dagresult: ProtocolDAGResult = execute_DAG(
                dag, shared_basedir=shared, scratch_basedir=scratch
            )

        return protocol_short, dag, dagresult

    @pytest.fixture
    def protocol_dag_invalid_mapping(
        self,
        protocol_short,
        benzene_vacuum_system,
        toluene_vacuum_system,
        broken_mapping,
        tmpdir,
    ):
        dag = protocol_short.create(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            name="Broken vacuum transformation",
            mapping=broken_mapping,
        )
        with tmpdir.as_cwd():
            shared = Path("shared")
            shared.mkdir()

            scratch = Path("scratch")
            scratch.mkdir()

            # Don't raise the error for getting ProtocolResult
            dagresult: ProtocolDAGResult = execute_DAG(
                dag, raise_error=False, shared_basedir=shared, scratch_basedir=scratch
            )

        return protocol_short, dag, dagresult

    def test_dag_execute(self, protocol_dag_result):
        protocol, dag, dagresult = protocol_dag_result

        assert dagresult.ok()

        # the FinishUnit will always be the last to execute
        finishresult = dagresult.protocol_unit_results[-1]
        assert finishresult.name == "result"

    def test_terminal_units(self, protocol_dag_result):
        prot, dag, res = protocol_dag_result

        finals = res.terminal_protocol_unit_results

        assert len(finals) == 1
        assert isinstance(finals[0], ProtocolUnitResult)
        assert finals[0].name == "result"

    # TODO: We probably need to find failure test cases as control
    # def test_dag_execute_failure(self, protocol_dag_broken):
    #     protocol, dag, dagfailure = protocol_dag_broken
    #
    #     assert not dagfailure.ok()
    #     assert isinstance(dagfailure, ProtocolDAGResult)
    #
    #     failed_units = dagfailure.protocol_unit_failures
    #
    #     assert len(failed_units) == 1
    #     assert isinstance(failed_units[0], ProtocolUnitFailure)
    #
    # def test_dag_execute_failure_raise_error(
    #     self,
    #     protocol_short,
    #     benzene_vacuum_system,
    #     toluene_vacuum_system,
    #     broken_mapping,
    #     tmpdir,
    # ):
    #     """Executes a bad setup of a protocol DAG which has an incorrect mapping"""
    #     dag = protocol_short.create(
    #         stateA=benzene_vacuum_system,
    #         stateB=toluene_vacuum_system,
    #         name="a broken dummy run",
    #         mapping=broken_mapping,
    #     )
    #
    #     # tries to access an atom index that does not exist
    #     with tmpdir.as_cwd():
    #         shared = Path("shared")
    #         shared.mkdir()
    #
    #         scratch = Path("scratch")
    #         scratch.mkdir()
    #
    #         with pytest.raises(IndexError):
    #             execute_DAG(
    #                 dag,
    #                 raise_error=True,
    #                 shared_basedir=shared,
    #                 scratch_basedir=scratch,
    #             )

    def test_create_with_invalid_mapping(
        self,
        protocol_short_multiple_cycles,
        benzene_solvent_system,
        toluene_solvent_system,
        mapping_benzonitrile_styrene,
    ):
        """
        Attempt creating a protocol with an invalid mapping. Components in mapping don't
        match the components in the states/systems.

        We expect it to fail with an exception.
        """
        protocol = protocol_short_multiple_cycles

        with pytest.raises(AssertionError):
            _ = protocol.create(
                stateA=benzene_solvent_system,
                stateB=toluene_solvent_system,
                name="Short solvent transformation",
                mapping=mapping_benzonitrile_styrene,
            )

    def test_create_with_invalid_componentA_mapping(
        self,
        protocol_short_multiple_cycles,
        benzene_solvent_system,
        styrene_solvent_system,
        mapping_benzonitrile_styrene,
    ):
        """
        Test creating a protocol with the componentA of the mapping not matching the given
        component in stateA.

        We expect it to fail with an exception.
        """
        protocol = protocol_short_multiple_cycles

        with pytest.raises(AssertionError):
            _ = protocol.create(
                stateA=benzene_solvent_system,
                stateB=styrene_solvent_system,
                name="Short solvent transformation",
                mapping=mapping_benzonitrile_styrene,
            )

    def test_create_with_invalid_componentB_mapping(
        self,
        protocol_short_multiple_cycles,
        benzonitrile_solvent_system,
        toluene_solvent_system,
        mapping_benzonitrile_styrene,
    ):
        """
        Test creating a protocol with the componentB of the mapping not matching the given
        component in stateB.

        We expect it to fail with an exception.
        """
        protocol = protocol_short_multiple_cycles

        with pytest.raises(AssertionError):
            _ = protocol.create(
                stateA=benzonitrile_solvent_system,
                stateB=toluene_solvent_system,
                name="Short solvent transformation",
                mapping=mapping_benzonitrile_styrene,
            )

    @pytest.mark.parametrize(
        "protocol",
        [
            "protocol_short_multiple_cycles",
            #'protocol_short_multiple_cycles_gpu'
        ],
    )
    def test_create_execute_gather(
        self,
        protocol,
        benzene_vacuum_system,
        toluene_vacuum_system,
        mapping_benzene_toluene,
        tmpdir,
        request,
    ):
        """
        Perform 20 independent simulations of the NEQ cycling protocol for the benzene to toluene
        transformation and gather the results.

        This is done by using 4 replicates of the protocol with 5 simulation units each.
        """
        import numpy as np

        protocol = request.getfixturevalue(protocol)

        dag = protocol.create(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            name="Short vacuum transformation",
            mapping=mapping_benzene_toluene,
        )

        results = []
        n_replicates = 4
        for i in range(n_replicates):
            with tmpdir.as_cwd():
                shared = Path(f"shared_{i}")
                shared.mkdir()

                scratch = Path(f"scratch_{i}")
                scratch.mkdir()

                dagresult = execute_DAG(
                    dag, shared_basedir=shared, scratch_basedir=scratch
                )
                results.append(dagresult)
        # gather aggregated results of interest
        protocolresult = protocol.gather(results)

        # Check that it runs without failures
        for dag_result in results:
            failed_units = dag_result.protocol_unit_failures
            assert len(failed_units) == 0, "Unit failure in protocol dag result."

        # Get an estimate that is not NaN
        fe_estimate = protocolresult.get_estimate()
        fe_error = protocolresult.get_uncertainty()
        assert not np.isnan(fe_estimate), "Free energy estimate is NaN."
        assert not np.isnan(fe_error), "Free energy error estimate is NaN."
        # print(f"Free energy = {fe_estimate} +/- {fe_error}") # DEBUG

    @pytest.mark.skip(
        reason="Ambertools failing to parameterize. Review when we have full nagl."
    )
    @pytest.mark.gpu_ci
    @pytest.mark.parametrize(
        "protocol",
        [
            "protocol_short_multiple_cycles",
            #'protocol_short_multiple_cycles_gpu'
        ],
    )
    def test_create_execute_gather_toluene_to_toluene(
        self,
        protocol,
        toluene_vacuum_system,
        mapping_toluene_toluene,
        tmpdir,
        request,
        toluene,
    ):
        """
        Perform 20 independent simulations of the NEQ cycling protocol for the toluene to toluene
        transformation and gather the results.

        This sets up a toluene to toluene transformation using the benzene to toluene mapping
        and check that the free energy estimates are around 0, within 6*dDG.

        This is done by using 4 repeats of the protocol with 5 simulation units each.

        Notes
        -----
        The error estimate for the free energy calculations is tried up to 5 times in case there
        are stochastic errors with the BAR calculations.

        This test is prone to fail on GPU. Numerical precision issues?
        """
        import numpy as np

        protocol = request.getfixturevalue(protocol)
        # rename the components
        toluene_state_a = toluene_vacuum_system.copy_with_replacements(
            components={"ligand": toluene.copy_with_replacements(name="toluene_a")}
        )
        toluene_state_b = toluene_vacuum_system.copy_with_replacements(
            components={"ligand": toluene.copy_with_replacements(name="toluene_b")}
        )
        dag = protocol.create(
            stateA=toluene_state_a,
            stateB=toluene_state_b,
            name="Toluene vacuum transformation",
            mapping=mapping_toluene_toluene,
        )

        results = []
        n_repeats = 4
        for i in range(n_repeats):
            with tmpdir.as_cwd():
                shared = Path(f"shared_{i}")
                shared.mkdir()

                scratch = Path(f"scratch_{i}")
                scratch.mkdir()

                dagresult = execute_DAG(
                    dag, shared_basedir=shared, scratch_basedir=scratch
                )
            results.append(dagresult)
        # gather aggregated results of interest
        protocolresult = protocol.gather(results)

        # Check that it runs without failures
        for dag_result in results:
            failed_units = dag_result.protocol_unit_failures
            assert len(failed_units) == 0, "Unit failure in protocol dag result."

        # Get an estimate that is not NaN
        fe_estimate = protocolresult.get_estimate()
        assert not np.isnan(fe_estimate), "Free energy estimate is NaN."

        # Test that estimate is around 0 within tolerance
        assert np.isclose(
            fe_estimate.magnitude, 0.0, atol=1e-10
        ), f"Free energy estimate {fe_estimate} is not close to zero."

        # Get an uncertainty; if it gives a BoundsError this isn't that
        # surprising given our values are so close to zero, so we'll allow it
        try:
            fe_error = protocolresult.get_uncertainty(n_bootstraps=100)
            assert not np.isnan(fe_error), "Free energy error estimate is NaN."
        except pymbar.utils.BoundsError as pymbar_error:
            pass

    # TODO: We could also generate a plot with the forward and reverse works and visually check the results.

    @pytest.mark.slow
    def test_tyk2_complex(
        self,
        protocol_short,
        tyk2_lig_ejm_54_complex,
        tyk2_lig_ejm_46_complex,
        mapping_tyk2_54_to_46,
        tmpdir,
    ):
        """
        Run the protocol with single transformation between ligands ejm_54 and ejm_46
        from the tyk2 dataset.
        """
        dag = protocol_short.create(
            stateA=tyk2_lig_ejm_54_complex,
            stateB=tyk2_lig_ejm_46_complex,
            name="Short protein-ligand complex transformation",
            mapping=mapping_tyk2_54_to_46,
        )

        with tmpdir.as_cwd():
            shared = Path("shared")
            shared.mkdir()
            scratch = Path("scratch")
            scratch.mkdir()

            dagresult = execute_DAG(
                dag,
                shared_basedir=shared,
                scratch_basedir=scratch,
            )

        # Check that the dag was executed correctly
        assert dagresult.ok(), f"DAG was not executed correctly."

    @pytest.mark.parametrize("method, backend", sorted(partial_charges_config()))
    def test_partial_charge_assignation(
        self,
        short_settings,
        benzene_vacuum_system,
        toluene_vacuum_system,
        mapping_benzene_toluene,
        method,
        backend,
        tmpdir,
    ):
        """
        Test the different options for method and backend for partial charge assignation produces
        successful protocol runs.
        """
        # Deep copy of settings to modify
        local_settings = short_settings.copy(deep=True)
        local_settings.partial_charge_settings.partial_charge_method = method
        local_settings.partial_charge_settings.off_toolkit_backend = backend

        protocol = NonEquilibriumCyclingProtocol(settings=local_settings)

        dag = protocol.create(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            name="Short vacuum transformation",
            mapping=mapping_benzene_toluene,
        )

        with tmpdir.as_cwd():
            shared = Path("shared")
            shared.mkdir()

            scratch = Path("scratch")
            scratch.mkdir()

            dagresult: ProtocolDAGResult = execute_DAG(
                dag, shared_basedir=shared, scratch_basedir=scratch
            )

        assert dagresult.ok()

    @pytest.mark.parametrize(
        "method, backend", [("am1bcc", "rdkit"), ("am1bccelf10", "ambertools")]
    )
    def test_failing_partial_charge_assign(
        self,
        short_settings,
        benzene_vacuum_system,
        toluene_vacuum_system,
        mapping_benzene_toluene,
        method,
        backend,
        tmpdir,
    ):
        """
        Test that incompatible method and backend combinations for partial charge assignation.
        We expect a ``ValueError`` exception to be raised in these cases.
        """
        # Deep copy of settings to modify
        local_settings = short_settings.copy(deep=True)
        local_settings.partial_charge_settings.partial_charge_method = method
        local_settings.partial_charge_settings.off_toolkit_backend = backend

        protocol = NonEquilibriumCyclingProtocol(settings=local_settings)

        dag = protocol.create(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            name="Short vacuum transformation",
            mapping=mapping_benzene_toluene,
        )

        with tmpdir.as_cwd():
            with pytest.raises(ValueError):
                shared = Path("shared")
                shared.mkdir()

                scratch = Path("scratch")
                scratch.mkdir()

                execute_DAG(dag, shared_basedir=shared, scratch_basedir=scratch)

    def test_fail_with_multiple_solvent_comps(
        self,
        protocol_short,
        benzene_solvent_system,
        toluene_double_solvent_system,
        mapping_benzene_toluene,
        tmpdir,
    ):
        with pytest.raises(AssertionError):
            _ = protocol_short.create(
                stateA=benzene_solvent_system,
                stateB=toluene_double_solvent_system,
                name="Broken double solvent transformation",
                mapping=mapping_benzene_toluene,
    def test_error_with_multiple_mappings(
        self,
        protocol_short,
        benzene_vacuum_system,
        toluene_vacuum_system,
        mapping_benzene_toluene,
    ):
        """
        Make sure that when a list of mappings is passed that an error is raised.
        """

        with pytest.raises(
            ValueError, match="A single LigandAtomMapping is expected for this Protocol"
        ):
            _ = protocol_short.create(
                stateA=benzene_vacuum_system,
                stateB=toluene_vacuum_system,
                name="Test protocol",
                mapping=[mapping_benzene_toluene, mapping_benzene_toluene],
            )


class TestSetupUnit:
    @pytest.fixture(scope="class")
    def tyk2_protein_comp(self):
        from gufe import ProteinComponent

        input_pdb = str(
            files("feflow.tests.data.protein_ligand").joinpath("tyk2_protein.pdb")
        )
        protein_comp = ProteinComponent.from_pdb_file(input_pdb)
        return protein_comp

    @pytest.fixture(scope="class")
    def tyk2_lig_ejm_31_comp(self):
        from gufe import SmallMoleculeComponent

        input_sdf = str(
            files("feflow.tests.data.protein_ligand").joinpath("tyk2_lig_ejm_31.sdf")
        )
        small_mol_comp = SmallMoleculeComponent.from_sdf_file(input_sdf)
        return small_mol_comp

    @pytest.fixture(scope="class")
    def tyk2_lig_ejm_55_comp(self):
        from gufe import SmallMoleculeComponent

        input_sdf = str(
            files("feflow.tests.data.protein_ligand").joinpath("tyk2_lig_ejm_55.sdf")
        )
        small_mol_comp = SmallMoleculeComponent.from_sdf_file(input_sdf)
        return small_mol_comp

    @pytest.fixture(scope="class")
    def tyk2_lig_ejm_31_to_lig_ejm_55_mapping(
        self, tyk2_lig_ejm_31_comp, tyk2_lig_ejm_55_comp
    ):
        from kartograf import KartografAtomMapper

        atom_mapper = KartografAtomMapper()
        mapping = next(
            atom_mapper.suggest_mappings(tyk2_lig_ejm_31_comp, tyk2_lig_ejm_55_comp)
        )
        return mapping

    @pytest.fixture(scope="class")
    def tyk2_ejm_31_to_ejm_55_systems_only_ligands(
        self, tyk2_lig_ejm_31_to_lig_ejm_55_mapping, solvent_comp
    ):
        """
        This fixture returns a dictionary with both state A and state B for the tyk2
        lig_ejm_31 to lig_ejm_55 transformation, as chemical systems. The systems are solvated.
        """
        from gufe import ChemicalSystem

        state_a = {
            "ligand": tyk2_lig_ejm_31_to_lig_ejm_55_mapping.componentA,
            "solvent": solvent_comp,
        }
        state_b = {
            "ligand": tyk2_lig_ejm_31_to_lig_ejm_55_mapping.componentB,
            "solvent": solvent_comp,
        }
        system_a = ChemicalSystem(state_a)
        system_b = ChemicalSystem(state_b)
        return {"state_a": system_a, "state_b": system_b}

    @pytest.fixture(scope="class")
    def tyk2_ejm_31_to_ejm_55_systems(
        self, tyk2_protein_comp, tyk2_lig_ejm_31_to_lig_ejm_55_mapping, solvent_comp
    ):
        """
        This fixture returns a dictionary with both state A and state B for the tyk2
        lig_ejm_31 to lig_ejm_55 transformation, as chemical systems. The systems are solvated.
        """
        from gufe import ChemicalSystem

        state_a = {
            "protein": tyk2_protein_comp,
            "ligand": tyk2_lig_ejm_31_to_lig_ejm_55_mapping.componentA,
            "solvent": solvent_comp,
        }
        state_b = {
            "protein": tyk2_protein_comp,
            "ligand": tyk2_lig_ejm_31_to_lig_ejm_55_mapping.componentB,
            "solvent": solvent_comp,
        }
        system_a = ChemicalSystem(state_a)
        system_b = ChemicalSystem(state_b)
        return {"state_a": system_a, "state_b": system_b}

    def test_setup_user_charges(
        self, benzene_modifications, mapping_benzene_toluene, tmpdir
    ):
        """
        Tests that the charges assigned to the topology are not changed along the way, respecting
        charges given by the users.

        This test inspects the results of the SetupUnit in the protocol.

        It setups a benzene to toluene transformation by first manually assigning partial charges
        to the ligands before creating the chemical systems, and checking that the generated charges
        are as expected after setting up the simulation.
        """
        import numpy as np
        from openff.toolkit import Molecule
        from openff.units import unit
        from gufe import ChemicalSystem, Context, LigandAtomMapping
        from gufe.components import SmallMoleculeComponent
        from feflow.protocols.nonequilibrium_cycling import SetupUnit

        def _assign_random_partial_charges(offmol: Molecule, seed: int = 42):
            """
            Assign random partial charges to given molecule such that the sum of the partial charges
            equals to the net formal charge of the molecule. This is done by generating random
            numbers up to n_atoms - 1 and determining the last one such that it complies with the
            formal charge restriction.
            """
            rng = np.random.default_rng(seed)  # Local seeded RNG
            total_charge = offmol.total_charge.m  # magnitude of formal charge
            partial_charges = rng.uniform(low=-0.1, high=0.1, size=offmol.n_atoms - 1)
            charge_diff = total_charge - np.sum(partial_charges)
            partial_charges = np.append(partial_charges, charge_diff)
            offmol.partial_charges = partial_charges * unit.elementary_charge
            return partial_charges

        benzene = Molecule.from_rdkit(benzene_modifications["benzene"])
        toluene = Molecule.from_rdkit(benzene_modifications["toluene"])
        # Forcing assignment of partial charges
        benzene_orig_charges = _assign_random_partial_charges(benzene)
        toluene_orig_charges = _assign_random_partial_charges(toluene)

        small_comp_a = SmallMoleculeComponent.from_openff(benzene)
        small_comp_b = SmallMoleculeComponent.from_openff(toluene)

        # IMPORTANT: We need to regenerate mapping because the underlying components changed
        # when we added the charges.
        mapping = LigandAtomMapping(
            componentA=small_comp_a,
            componentB=small_comp_b,
            componentA_to_componentB=mapping_benzene_toluene.componentA_to_componentB,
            annotations=mapping_benzene_toluene.annotations,
        )

        state_a = ChemicalSystem({"ligand": small_comp_a})
        state_b = ChemicalSystem({"ligand": small_comp_b})

        settings = NonEquilibriumCyclingProtocol.default_settings()
        # Make sure to use CPU platform for tests
        settings.engine_settings.compute_platform = "CPU"
        protocol = NonEquilibriumCyclingProtocol(settings=settings)

        setup = SetupUnit(
            state_a=state_a,
            state_b=state_b,
            mapping=mapping,
            protocol=protocol,
            name="setup_user_charges",
        )

        # Run unit and extract results
        scratch_path = Path(tmpdir / "scratch")
        shared_path = Path(tmpdir / "shared")
        scratch_path.mkdir()
        shared_path.mkdir()
        context = Context(scratch=scratch_path, shared=shared_path)
        setup_result = setup.execute(context=context, **setup.inputs)
        with open(setup_result.outputs["topology_path"], "rb") as in_file:
            htf = pickle.load(in_file)

        # Finally check that the charges are as expected
        _check_htf_charges(htf, benzene_orig_charges, toluene_orig_charges)

    def test_solvent_phase_tyk2_setup(
        self,
        tyk2_ejm_31_to_ejm_55_systems_only_ligands,
        tyk2_lig_ejm_31_to_lig_ejm_55_mapping,
        tmpdir,
    ):
        """
        Test setup of a solvent leg/phase for a protein-ligand simulation with TYK2 system and
        a specific transformation that has "challenging" atom mapping.
        """
        from feflow.protocols.nonequilibrium_cycling import SetupUnit
        from gufe import Context

        state_a = tyk2_ejm_31_to_ejm_55_systems_only_ligands["state_a"]
        state_b = tyk2_ejm_31_to_ejm_55_systems_only_ligands["state_b"]
        mapping = tyk2_lig_ejm_31_to_lig_ejm_55_mapping

        settings = NonEquilibriumCyclingProtocol.default_settings()
        # make sure to use CPU platform for tests
        settings.engine_settings.compute_platform = "CPU"
        # Using openeye partial charges seems to behave more stably than default ambertools
        settings.partial_charge_settings.off_toolkit_backend = "openeye"
        protocol = NonEquilibriumCyclingProtocol(settings=settings)

        setup = SetupUnit(
            state_a=state_a,
            state_b=state_b,
            mapping=mapping,
            protocol=protocol,
            name="setup_user_charges",
        )

        # Run unit and extract results
        scratch_path = Path(tmpdir / "scratch")
        shared_path = Path(tmpdir / "shared")
        scratch_path.mkdir()
        shared_path.mkdir()
        context = Context(scratch=scratch_path, shared=shared_path)

        # TODO: raising error here and the following assertion seem redundant
        setup_result = setup.execute(context=context, **setup.inputs, raise_error=True)

        assert setup_result.ok(), "Setup unit did not run successfully."

    @pytest.mark.gpu_ci
    def test_protein_ligand_tyk2_setup(
        self,
        tyk2_ejm_31_to_ejm_55_systems,
        tyk2_lig_ejm_31_to_lig_ejm_55_mapping,
        tmpdir,
    ):
        """
        Test setup of a production-like protein-ligand simulation with TYK2 system and
        a specific transformation that has "challenging" atom mapping.
        """
        from feflow.protocols.nonequilibrium_cycling import SetupUnit
        from gufe import Context

        state_a = tyk2_ejm_31_to_ejm_55_systems["state_a"]
        state_b = tyk2_ejm_31_to_ejm_55_systems["state_b"]
        mapping = tyk2_lig_ejm_31_to_lig_ejm_55_mapping

        settings = NonEquilibriumCyclingProtocol.default_settings()
        # make sure to use CPU platform for tests
        settings.engine_settings.compute_platform = "CPU"
        # Using openeye partial charges seems to behave more stably than default ambertools
        settings.partial_charge_settings.off_toolkit_backend = "openeye"
        protocol = NonEquilibriumCyclingProtocol(settings=settings)

        setup = SetupUnit(
            state_a=state_a,
            state_b=state_b,
            mapping=mapping,
            protocol=protocol,
            name="setup_user_charges",
        )

        # Run unit and extract results
        scratch_path = Path(tmpdir / "scratch")
        shared_path = Path(tmpdir / "shared")
        scratch_path.mkdir()
        shared_path.mkdir()
        context = Context(scratch=scratch_path, shared=shared_path)

        # TODO: raising error here and the following assertion seem redundant
        setup_result = setup.execute(context=context, **setup.inputs, raise_error=True)

        assert setup_result.ok(), "Setup unit did not run successfully."

def test_settings_round_trip():
    """
    Make sure we can round trip the settings class to and from json,
    related to <https://github.com/OpenFreeEnergy/feflow/issues/87>.
    """
    neq_settings = NonEquilibriumCyclingProtocol.default_settings()
    neq_json = json.dumps(neq_settings.model_dump(), cls=JSON_HANDLER.encoder)
    neq_settings_2 = NonEquilibriumCyclingSettings.model_validate(
        json.loads(neq_json, cls=JSON_HANDLER.decoder)
    )
    assert neq_settings == neq_settings_2
