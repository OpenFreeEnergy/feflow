"""Test related to the protein mutation protocol and utility functions"""

import json
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from gufe import ProteinComponent, ChemicalSystem, ProtocolDAGResult, LigandAtomMapping
from gufe.protocols.protocoldag import execute_DAG
from gufe.tokenization import JSON_HANDLER
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
def arg_capped():
    """ProteinComponent for Arginine residue capped by ACE and NME.
    This is meant to be used for testing charge transformations."""
    input_pdb = str(files("feflow.tests.data.capped_AAs").joinpath("ARG_capped.pdb"))
    protein_comp = ProteinComponent.from_pdb_file(input_pdb)
    return protein_comp


@pytest.fixture(scope="session")
def lys_capped():
    """ProteinComponent for Lysine residue capped by ACE and NME.
    This is meant to be used for testing charge transformations."""
    input_pdb = str(files("feflow.tests.data.capped_AAs").joinpath("LYS_capped.pdb"))
    protein_comp = ProteinComponent.from_pdb_file(input_pdb)
    return protein_comp


@pytest.fixture(scope="session")
def pro_capped():
    """ProteinComponent for Proline residue capped by ACE and NME."""
    input_pdb = str(files("feflow.tests.data.capped_AAs").joinpath("PRO_capped.pdb"))
    protein_comp = ProteinComponent.from_pdb_file(input_pdb)
    return protein_comp


@pytest.fixture(scope="session")
def glu_capped():
    """ProteinComponent for Glutamic Acid residue capped by ACE and NME."""
    input_pdb = str(files("feflow.tests.data.capped_AAs").joinpath("GLU_capped.pdb"))
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
def arg_capped_system(arg_capped, solvent_comp):
    """Solvated capped Arginine ChemicalSystem"""
    return ChemicalSystem({"protein": arg_capped, "solvent": solvent_comp})


@pytest.fixture(scope="session")
def lys_capped_system(lys_capped, solvent_comp):
    """Solvated capped Lysine ChemicalSystem"""
    return ChemicalSystem({"protein": lys_capped, "solvent": solvent_comp})


@pytest.fixture(scope="session")
def pro_capped_system(pro_capped, solvent_comp):
    """Solvated capped Proline ChemicalSystem"""
    return ChemicalSystem({"protein": pro_capped, "solvent": solvent_comp})


@pytest.fixture(scope="session")
def glu_capped_system(glu_capped, solvent_comp):
    """Solvated capped Glutamic Acid ChemicalSystem"""
    return ChemicalSystem({"protein": glu_capped, "solvent": solvent_comp})


@pytest.fixture(scope="session")
def ala_to_gly_mapping():
    """Mapping from ALA to GLY (capped)"""
    input_file = str(
        files("feflow.tests.data.capped_AAs").joinpath("ala_to_gly_mapping.json")
    )
    with open(input_file) as in_file:
        mapping = LigandAtomMapping.from_dict(
            json.load(in_file, cls=JSON_HANDLER.decoder)
        )
    return mapping


@pytest.fixture(scope="session")
def gly_to_ala_mapping(ala_capped, gly_capped, ala_to_gly_mapping):
    """GLY to ALA mapping. Inverts the ala_to_gly_mapping fixture."""
    gly_to_ala_map = ala_to_gly_mapping.componentB_to_componentA
    mapping = LigandAtomMapping(
        componentA=gly_capped,
        componentB=ala_capped,
        componentA_to_componentB=gly_to_ala_map,
    )
    return mapping


@pytest.fixture(scope="session")
def ala_to_arg_mapping():
    """Mapping from ALA to ARG (capped). Positive charge transformation."""
    input_file = str(
        files("feflow.tests.data.capped_AAs").joinpath("ala_to_arg_mapping.json")
    )
    with open(input_file) as in_file:
        mapping = LigandAtomMapping.from_dict(
            json.load(in_file, cls=JSON_HANDLER.decoder)
        )
    return mapping


@pytest.fixture(scope="session")
def arg_to_ala_mapping(ala_capped, arg_capped, ala_to_arg_mapping):
    """ARG to ALA mapping. Inverts the ala_to_arg_mapping fixture."""
    arg_to_ala_map = ala_to_arg_mapping.componentB_to_componentA
    mapping = LigandAtomMapping(
        componentA=arg_capped,
        componentB=ala_capped,
        componentA_to_componentB=arg_to_ala_map,
    )
    return mapping


@pytest.fixture(scope="session")
def ala_to_lys_mapping():
    """Mapping from ALA to LYS (capped)."""
    input_file = str(
        files("feflow.tests.data.capped_AAs").joinpath("ala_to_lys_mapping.json")
    )
    with open(input_file) as in_file:
        mapping = LigandAtomMapping.from_dict(
            json.load(in_file, cls=JSON_HANDLER.decoder)
        )
    return mapping


@pytest.fixture(scope="session")
def lys_to_ala_mapping(ala_capped, lys_capped, ala_to_lys_mapping):
    """GLY to ALA mapping. Inverts the ala_to_gly_mapping fixture."""
    lys_to_ala_map = ala_to_lys_mapping.componentB_to_componentA
    mapping = LigandAtomMapping(
        componentA=lys_capped,
        componentB=ala_capped,
        componentA_to_componentB=lys_to_ala_map,
    )
    return mapping


@pytest.fixture(scope="session")
def asp_to_leu_mapping(asp_capped, leu_capped):
    """Mapping from ASP to LEU (capped). Charge transformation."""
    input_file = str(
        files("feflow.tests.data.capped_AAs").joinpath("asp_to_leu_mapping.json")
    )
    with open(input_file) as in_file:
        mapping = LigandAtomMapping.from_dict(
            json.load(in_file, cls=JSON_HANDLER.decoder)
        )
    return mapping


@pytest.fixture(scope="session")
def ala_to_pro_mapping(ala_capped, pro_capped):
    """Mapping from ALA to PRO (capped). Ring breaking challenging transformation."""
    input_file = str(
        files("feflow.tests.data.capped_AAs").joinpath("ala_to_pro_mapping.json")
    )
    with open(input_file) as in_file:
        mapping = LigandAtomMapping.from_dict(json.load(in_file))
    return mapping


@pytest.fixture(scope="session")
def lys_to_glu_mapping(lys_capped, glu_capped):
    """Mapping from LYS to GLU (capped). Double charge-changing transformation."""
    input_file = str(
        files("feflow.tests.data.capped_AAs").joinpath("lys_to_glu_mapping.json")
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
            stateA=asp_capped_system,
            stateB=leu_capped_system,
            name="Short vacuum transformation",
            mapping=asp_to_leu_mapping,
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

    def execute_forward_reverse_dag(
        self, component_a, component_b, system_a, system_b, mapping_obj, tmpdir
    ):
        """
        Perform a forward and reverse free energy (FE) estimation between two protein mutation systems.

        This function creates and executes forward and reverse directed acyclic graphs (DAGs) for
        computing the free energy difference between two protein systems, typically involving
        a mutation (e.g., Arginine to Alanine). It calculates the forward and reverse free energy
        estimates, and checks that their sum is close to zero (indicating a consistent reversible
        transformation).

        Parameters
        ----------
        component_a: ProteinComponent
        component_b: ProteinComponent
        system_a: ChemicalSystem
        system_b: ChemicalSystem
        mapping_obj: LigandAtomMapping
        tmpdir

        Returns
        -------
        dict
            A dictionary containing FE estimates and uncertainties for both forward and reverse
            transformations:
            `{"forward": (forward_fe, forward_error), "reverse": (reverse_fe, reverse_error)}`.
        """
        settings = ProteinMutationProtocol.default_settings()
        protocol = ProteinMutationProtocol(settings=settings)

        forward_dag = protocol.create(
            stateA=system_a,
            stateB=system_b,
            name="Short vacuum transformation",
            mapping=mapping_obj,
        )
        # Reverse mapping
        map_dict = mapping_obj.componentB_to_componentA
        mapping = LigandAtomMapping(
            componentA=component_b,
            componentB=component_a,
            componentA_to_componentB=map_dict,
        )
        reverse_dag = protocol.create(
            stateA=system_b,
            stateB=system_a,
            name="Short vacuum transformation",
            mapping=mapping,
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

        data_obj = {
            "forward": (forward_fe, forward_error),
            "reverse": (reverse_fe, reverse_error),
        }

        return data_obj

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
    def test_ala_gly_convergence(
        self,
        ala_capped,
        gly_capped,
        ala_capped_system,
        gly_capped_system,
        ala_to_gly_mapping,
        gly_to_ala_mapping,
        tmpdir,
    ):
        """Convergence test for ALA to GLY forward and reverse neutral protein mutation protocol
        execution with default (production-ready) settings. Runs ALA to GLY and compares the
        FE estimate with running GLY to ALA."""

        results = self.execute_forward_reverse_dag(
            ala_capped,
            gly_capped,
            ala_capped_system,
            gly_capped_system,
            ala_to_gly_mapping,
            tmpdir,
        )

        # they should add up to close to zero
        forward_reverse_sum = abs(results["forward"][0] + results["reverse"][0])
        forward_reverse_sum_err = np.sqrt(
            results["forward"][1] ** 2 + results["reverse"][1] ** 2
        )

        print(
            f"DDG: {forward_reverse_sum}, 6*dDDG: {6 * forward_reverse_sum_err}"
        )  # DEBUG
        assert forward_reverse_sum < 6 * forward_reverse_sum_err, (
            f"DDG ({forward_reverse_sum}) is greater than "
            f"6 * dDDG ({6 * forward_reverse_sum_err})"
        )

    @pytest.mark.slow
    def test_charge_changing_convergence(
        self,
        arg_capped,
        ala_capped,
        lys_capped,
        ala_capped_system,
        arg_capped_system,
        lys_capped_system,
        arg_to_ala_mapping,
        lys_to_ala_mapping,
        tmpdir,
    ):
        """
        Test for charge changing transformation for the protein mutation protocol.

        We perform two forward and reverse and check the mutual convergence.
            Positive: ARG -> ALA -> ARG
            Negative: LYS -> ALA -> LYS

        The need to do it this way is because since we are introducing counterions the energies for
        each will not match, therefore to cancel this contribution we compare between both positive
        and negative full cycles.
        """
        # Create and execute DAGs
        arg_results = self.execute_forward_reverse_dag(
            arg_capped,
            ala_capped,
            arg_capped_system,
            ala_capped_system,
            arg_to_ala_mapping,
            tmpdir,
        )
        lys_results = self.execute_forward_reverse_dag(
            lys_capped,
            ala_capped,
            lys_capped_system,
            ala_capped_system,
            lys_to_ala_mapping,
            tmpdir,
        )

        # they should add up to close to zero
        arg_forward_reverse_sum = arg_results["forward"][0] + arg_results["reverse"][0]
        arg_forward_reverse_sum_error = (
            arg_results["forward"][1] ** 2 + arg_results["reverse"][1] ** 2
        )
        lys_forward_reverse_sum = lys_results["forward"][0] + lys_results["reverse"][0]
        lys_forward_reverse_sum_error = (
            lys_results["forward"][1] ** 2 + lys_results["reverse"][1] ** 2
        )

        # FE estimates are the first element, errors are the second element in the tuple
        arg_lys_diff = abs(arg_forward_reverse_sum - lys_forward_reverse_sum)
        arg_lys_diff_error = np.sqrt(
            arg_forward_reverse_sum_error + lys_forward_reverse_sum_error
        )

        print(
            f"DDG: {arg_lys_diff}, 6*dDDG: {6 * arg_lys_diff_error}"
        )  # debug control print
        assert arg_lys_diff < 6 * arg_lys_diff_error, (
            f"DDG ({arg_lys_diff}) is greater than "
            f"6 * dDDG ({6 * arg_lys_diff_error})"
        )

    def test_proline_mutation_fails(
        self, ala_capped_system, pro_capped_system, ala_to_pro_mapping
    ):
        """Test that attempting to make a protein mutation that involves proline (or ring breaking
        transformations) is not handled and results in an error.

        This test ensures that the mutation protocol correctly identifies mutations involving
        proline, which is typically a ring-breaking transformation that is not supported, and
        raises an appropriate error.

        Parameters
        ----------
        ala_capped_system : ChemicalSystem
            The chemical system representing a capped alanine residue.
        pro_capped_system : ChemicalSystem
            The chemical system representing a capped proline residue.
        ala_to_pro_mapping : LigandAtomMapping
            Mapping object representing the atom mapping from ALA to PRO.
        """
        from feflow.utils.exceptions import MethodLimitationtError

        settings = ProteinMutationProtocol.default_settings()
        protocol = ProteinMutationProtocol(settings=settings)

        # Expect an error when trying to create the DAG with this invalid transformation
        with pytest.raises(MethodLimitationtError, match="proline.*not supported"):
            protocol.create(
                stateA=ala_capped_system,
                stateB=pro_capped_system,
                name="Invalid proline mutation",
                mapping=ala_to_pro_mapping,
            )

    def test_double_charge_fails(
        self, lys_capped_system, glu_capped_system, lys_to_glu_mapping
    ):
        """
        Test that attempting a mutation with a double charge change between lysine and glutamate
        systems raises a `NotSupportedError`.

        This test verifies that the `ProteinMutationProtocol` correctly raises an error when trying to
        create a directed acyclic graph (DAG) for an invalid mutation involving a double charge change.
        The test expects the `NotSupportedError` to be raised with a message indicating that
        double-charge transformations are not supported.

        Parameters
        ----------
        lys_capped_system : ChemicalSystem
            Molecular system with a capped lysine residue, representing the initial state (A).
        glu_capped_system : ChemicalSystem
            Molecular system with a capped glutamate residue, representing the target state (B).
        lys_to_glu_mapping : LigandAtomMapping
            Atom mapping defining the correspondence between atoms in the lysine and glutamate systems.
        """
        from feflow.utils.exceptions import ProtocolSupportError

        settings = ProteinMutationProtocol.default_settings()
        protocol = ProteinMutationProtocol(settings=settings)

        # Expect an error when trying to create the DAG with this invalid transformation
        with pytest.raises(ProtocolSupportError, match="double charge.*not supported"):
            protocol.create(
                stateA=lys_capped_system,
                stateB=glu_capped_system,
                name="Invalid proline mutation",
                mapping=lys_to_glu_mapping,
            )
