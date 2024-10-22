"""Test related to the protein mutation protocol and utility functions"""

import json
from importlib.resources import files
from pathlib import Path

import pytest
from gufe import ProteinComponent, ChemicalSystem, ProtocolDAGResult, LigandAtomMapping
from gufe.protocols.protocoldag import execute_DAG
from feflow.protocols import ProteinMutationProtocol


# Fixtures
@pytest.fixture(scope="session")
def ala_capped():
    """ProteinComponent for Alanine residue capped by ACE and NME."""
    input_pdb = str(files("feflow.tests.data.capped_AAs").joinpath("ALA_capped.pdb"))
    protein_comp = ProteinComponent.from_pdb_file(input_pdb)
    return protein_comp


@pytest.fixture(scope="session")
def gly_capped():
    """ProteinComponent for Glycine residue capped by ACE and NME"""
    input_pdb = str(files("feflow.tests.data.capped_AAs").joinpath("GLY_capped.pdb"))
    protein_comp = ProteinComponent.from_pdb_file(input_pdb)
    return protein_comp


@pytest.fixture(scope="session")
def asp_capped():
    """ProteinComponent for Aspartic Acid residue capped by ACE and NME.
    This is meant to be used for testing charge transformations."""
    input_pdb = str(files("feflow.tests.data.capped_AAs").joinpath("ASP_capped.pdb"))
    protein_comp = ProteinComponent.from_pdb_file(input_pdb)
    return protein_comp


@pytest.fixture(scope="session")
def leu_capped():
    """ProteinComponent for Leucine residue capped by ACE and NME.
    This is meant to be used for testing charge transformations."""
    input_pdb = str(files("feflow.tests.data.capped_AAs").joinpath("LEU_capped.pdb"))
    protein_comp = ProteinComponent.from_pdb_file(input_pdb)
    return protein_comp


@pytest.fixture(scope="session")
def ala_capped_system(ala_capped, solvent_comp):
    """Solvated capped Alanine ChemicalSystem"""
    return ChemicalSystem({"protein": ala_capped, "solvent": solvent_comp})


@pytest.fixture(scope="session")
def gly_capped_system(gly_capped, solvent_comp):
    """Solvated capped Alanine ChemicalSystem"""
    return ChemicalSystem({"protein": gly_capped, "solvent": solvent_comp})


@pytest.fixture(scope="session")
def asp_capped_system(asp_capped, solvent_comp):
    """Solvated capped Aspartic acid ChemicalSystem"""
    return ChemicalSystem({"protein": asp_capped, "solvent": solvent_comp})


@pytest.fixture(scope="session")
def leu_capped_system(leu_capped, solvent_comp):
    """Solvated capped Leucine ChemicalSystem"""
    return ChemicalSystem({"protein": leu_capped, "solvent": solvent_comp})


@pytest.fixture(scope="session")
def ala_to_gly_mapping():
    """Mapping from ALA to GLY (capped)"""
    input_file = str(
        files("feflow.tests.data.capped_AAs").joinpath("ala_to_gly_mapping.json")
    )
    with open(input_file) as in_file:
        mapping = LigandAtomMapping.from_dict(json.load(in_file))
    return mapping


@pytest.fixture(scope="session")
def gly_to_ala_mapping(ala_capped, gly_capped, ala_to_gly_mapping):
    """GLY to ALA mapping. Inverts the ala_to_gly_mapping fixture."""
    gly_to_ala_map = ala_to_gly_mapping.componentB_to_componentA
    mapping = LigandAtomMapping(componentA=gly_capped, componentB=ala_capped, componentA_to_componentB=gly_to_ala_map)
    return mapping


@pytest.fixture(scope="session")
def asp_to_leu_mapping(asp_capped, leu_capped):
    """Mapping from ASP to LEU (capped). Charge transformation."""
    from gufe import LigandAtomMapping

    input_file = str(
        files("feflow.tests.data.capped_AAs").joinpath("asp_to_leu_mapping.json")
    )
    with open(input_file) as in_file:
        mapping = LigandAtomMapping.from_dict(json.load(in_file))
    return mapping


class TestProtocolMutation:
    @pytest.fixture(scope="class")
    def short_settings_protein_mutation(self):
        settings = ProteinMutationProtocol.default_settings()

        settings.integrator_settings.equilibrium_steps = 1000
        settings.integrator_settings.nonequilibrium_steps = 1000
        settings.work_save_frequency = 50
        settings.traj_save_frequency = 250
        settings.num_cycles = 5
        settings.engine_settings.compute_platform = "CPU"

        return settings

    @pytest.fixture(scope="class")
    def protocol_short(self, short_settings_protein_mutation):
        return ProteinMutationProtocol(settings=short_settings_protein_mutation)

    @pytest.fixture(scope="class")
    def protocol_ala_to_gly_result(
        self,
        protocol_short,
        ala_capped_system,
        gly_capped_system,
        ala_to_gly_mapping,
        tmpdir,
    ):
        """Short protocol execution for capped ALA to GLY mutation"""
        dag = protocol_short.create(
            stateA=ala_capped_system,
            stateB=gly_capped_system,
            name="Short vacuum transformation",
            mapping=ala_to_gly_mapping,
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

    @pytest.fixture(scope="class")
    def protocol_asp_to_leu_result(
        self,
        protocol_short,
        asp_capped_system,
        leu_capped_system,
        asp_to_leu_mapping,
        tmpdir,
    ):
        """Short protocol execution for charge-changing mutation of capped ASP to LEU."""
        dag = protocol_short.create(
            stateA=ala_capped_system,
            stateB=gly_capped_system,
            name="Short vacuum transformation",
            mapping=ala_to_gly_mapping,
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

    def test_ala_to_gly_execute(self, protocol_ala_to_gly_result):
        """Takes a protocol result from an executed DAG and checks the OK status
        as well as the name of the resulting unit."""
        protocol, dag, dagresult = protocol_ala_to_gly_result

        assert dagresult.ok()

        # the FinishUnit will always be the last to execute
        finishresult = dagresult.protocol_unit_results[-1]
        assert finishresult.name == "result"

    def test_asp_to_leu_execute(self, protocol_asp_to_leu_result):
        """Takes a protocol result from an executed DAG and checks the OK status
        as well as the name of the resulting unit. Charge transformation case."""
        protocol, dag, dagresult = protocol_asp_to_leu_result

        assert dagresult.ok()

        # the FinishUnit will always be the last to execute
        finishresult = dagresult.protocol_unit_results[-1]
        assert finishresult.name == "result"

    @pytest.mark.slow
    def ala_gly_convergence(self, ala_capped_system, gly_capped_system, ala_to_gly_mapping, gly_to_ala_mapping):
        """Convergence test for ALA to GLY forward and reverse neutral protein mutation protocol
        execution with default (production-ready) settings. Runs ALA to GLY and compares the
        FE estimate with running GLY to ALA."""
        import numpy as np

        settings = ProteinMutationProtocol.default_settings()
        protocol = ProteinMutationProtocol(settings=settings)

        # Create forward and backward DAGs
        forward_dag = protocol.create(
            stateA=ala_capped_system,
            stateB=gly_capped_system,
            name="Short vacuum transformation",
            mapping=ala_to_gly_mapping,
        )
        reverse_dag = protocol.create(
            stateA=gly_capped_system,
            stateB=ala_capped_system,
            name="Short vacuum transformation",
            mapping=gly_to_ala_mapping,
        )

        # Execute DAGs
        with tmpdir.as_cwd():
            shared = Path("shared")
            shared.mkdir()

            scratch = Path("scratch")
            scratch.mkdir()

            forward_dagresult: ProtocolDAGResult = execute_DAG(
                forward_dag, shared_basedir=shared, scratch_basedir=scratch
            )
            reverse_dagresult: ProtocolDAGResult = execute_DAG(
                reverse_dag, shared_basedir=shared, scratch_basedir=scratch
            )

        # Verify DAGs were executed correctly
        assert forward_dagresult.ok()
        assert reverse_dagresult.ok()

        # Get FE estimate
        forward_fe = forward_dagresult.get_estimate()
        forward_error = forward_dagresult.get_uncertainty()
        reverse_fe = reverse_dagresult.get_estimate()
        reverse_error = reverse_dagresult.get_uncertainty()

        # they should add up to close to zero
        forward_reverse_sum = abs(forward_fe + reverse_fe)
        forward_reverse_sum_err = np.sqrt(
            forward_error ** 2 + reverse_error ** 2)
        print(f"DDG: {forward_reverse_sum}, 6*dDDG: {6 * forward_reverse_sum_err}")  # DEBUG
        assert forward_reverse_sum < 6 * forward_reverse_sum_err, (
            f"DDG ({forward_reverse_sum}) is greater than "
            f"6 * dDDG ({6 * forward_reverse_sum_err})")

    @pytest.mark.slow
    def test_charge_changing_convergence(self):
        """
        Test for charge changing transformation for the protein mutation protocol.

        We perform two forward and reverse and check the mutual convergence.
            Positive: ARG -> ALA -> ARG
            Negative: LYS -> ALA -> LYS

        The need to do it this way is because since we are introducing counterions the energies for
        each will not match, therefore to cancel this contribution we compare between both positive
        and negative full cycles.
        """
