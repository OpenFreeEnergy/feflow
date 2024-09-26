"""Test related to the protein mutation protocol and utility functions"""

import json
from importlib.resources import files
from pathlib import Path

import pytest
from gufe import ProteinComponent, ChemicalSystem, ProtocolDAGResult
from gufe.protocols.protocoldag import execute_DAG
from feflow.protocols import ProteinMutationProtocol


# Fixtures
@pytest.fixture(scope="module")
def ala_capped():
    """ProteinComponent for Alanine residue capped by ACE and NME."""
    input_pdb = str(files("feflow.tests.data").joinpath("ALA_capped.pdb"))
    protein_comp = ProteinComponent.from_pdb_file(input_pdb)
    return protein_comp


@pytest.fixture(scope="module")
def gly_capped():
    """ProteinComponent for Glycine residue capped by ACE and NME"""
    input_pdb = str(files("feflow.tests.data").joinpath("GLY_capped.pdb"))
    protein_comp = ProteinComponent.from_pdb_file(input_pdb)
    return protein_comp


@pytest.fixture(scope="module")
def asp_capped():
    """ProteinComponent for Aspartic Acid residue capped by ACE and NME.
    This is meant to be used for testing charge transformations."""
    input_pdb = str(files("feflow.tests.data").joinpath("ASP_capped.pdb"))
    protein_comp = ProteinComponent.from_pdb_file(input_pdb)
    return protein_comp


@pytest.fixture(scope="module")
def leu_capped():
    """ProteinComponent for Leucine residue capped by ACE and NME.
    This is meant to be used for testing charge transformations."""
    input_pdb = str(files("feflow.tests.data").joinpath("LEU_capped.pdb"))
    protein_comp = ProteinComponent.from_pdb_file(input_pdb)
    return protein_comp

@pytest.fixture(scope="module")
def ala_capped_system(ala_capped, solvent_comp):
    """Solvated capped Alanine ChemicalSystem"""
    return ChemicalSystem({"protein": ala_capped, "solvent": solvent_comp})


@pytest.fixture(scope="module")
def gly_capped_system(gly_capped, solvent_comp):
    """Solvated capped Alanine ChemicalSystem"""
    return ChemicalSystem({"protein": gly_capped, "solvent": solvent_comp})


@pytest.fixture(scope="module")
def asp_capped_system(asp_capped, solvent_comp):
    """Solvated capped Aspartic acid ChemicalSystem"""
    return ChemicalSystem({"protein": asp_capped, "solvent": solvent_comp})


@pytest.fixture(scope="module")
def leu_capped_system(leu_capped, solvent_comp):
    """Solvated capped Leucine ChemicalSystem"""
    return ChemicalSystem({"protein": leu_capped, "solvent": solvent_comp})


@pytest.fixture(scope="module")
def ala_to_gly_mapping(ala_capped, gly_capped):
    """Mapping from ALA to GLY (capped)"""
    from gufe import LigandAtomMapping
    input_file = str(files("feflow.tests.data").joinpath("ala_to_gly_mapping.json"))
    with open(input_file) as in_file:
        mapping = LigandAtomMapping.from_dict(json.load(in_file))
    return mapping


@pytest.fixture(scope="module")
def asp_to_leu_mapping(asp_capped, leu_capped):
    """Mapping from ASP to LEU (capped). Charge transformation."""
    from gufe import LigandAtomMapping
    input_file = str(files("feflow.tests.data").joinpath("asp_to_leu_mapping.json"))
    with open(input_file) as in_file:
        mapping = LigandAtomMapping.from_dict(json.load(in_file))
    return mapping


class TestProtocolMutation:
    @pytest.fixture(scope="class")
    def short_settings_protein_mutation(self):
        settings = ProteinMutationProtocol.default_settings()

        settings.integrator_settings.equlibrium_steps = 1000
        settings.integrator_settings.nonequlibrium_steps = 1000
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
