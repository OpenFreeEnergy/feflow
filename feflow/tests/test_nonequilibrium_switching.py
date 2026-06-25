import json
from pathlib import Path

import pytest

from feflow.protocols import (
    NonEquilibriumSwitchingProtocol,
    ForwardSwitchingUnit,
    ReverseSwitchingUnit,
)
from feflow.settings import NonEquilibriumSwitchingSettings
from gufe.protocols.protocoldag import ProtocolDAGResult, execute_DAG
from gufe.protocols.protocolunit import ProtocolUnitResult
from gufe.tokenization import JSON_HANDLER

# required plugins/fixtures
pytest_plugins = ["feflow.tests.fixtures.tyk2_fixtures"]


class TestNonEquilibriumSwitching:
    @pytest.fixture
    def protocol_short(self, short_switching_settings):
        return NonEquilibriumSwitchingProtocol(settings=short_switching_settings)

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
            name="Short vacuum switching",
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

    def test_dag_execute(self, protocol_dag_result):
        _, _, dagresult = protocol_dag_result
        assert dagresult.ok()
        assert dagresult.protocol_unit_results[-1].name == "result"

    def test_terminal_units(self, protocol_dag_result):
        _, _, res = protocol_dag_result
        finals = res.terminal_protocol_unit_results
        assert len(finals) == 1
        assert isinstance(finals[0], ProtocolUnitResult)
        assert finals[0].name == "result"

    def test_dag_has_forward_and_reverse_units(
        self,
        protocol_short,
        benzene_vacuum_system,
        toluene_vacuum_system,
        mapping_benzene_toluene,
    ):
        """DAG must contain both ForwardSwitchingUnit and ReverseSwitchingUnit instances."""
        dag = protocol_short.create(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            name="unit type check",
            mapping=mapping_benzene_toluene,
        )
        unit_types = [type(u) for u in dag.protocol_units]
        assert ForwardSwitchingUnit in unit_types
        assert ReverseSwitchingUnit in unit_types

    def test_num_switches_creates_correct_unit_counts(
        self,
        short_switching_settings,
        benzene_vacuum_system,
        toluene_vacuum_system,
        mapping_benzene_toluene,
    ):
        """num_switches forward + num_switches reverse + 1 setup + 1 result units."""
        protocol = NonEquilibriumSwitchingProtocol(settings=short_switching_settings)
        num_switches = short_switching_settings.num_switches
        dag = protocol.create(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            name="unit count check",
            mapping=mapping_benzene_toluene,
        )
        units = dag.protocol_units
        n_forward = sum(1 for u in units if isinstance(u, ForwardSwitchingUnit))
        n_reverse = sum(1 for u in units if isinstance(u, ReverseSwitchingUnit))
        assert n_forward == num_switches
        assert n_reverse == num_switches
        # +2 for SetupUnit and ResultUnit
        assert len(units) == 2 * num_switches + 2

    def test_create_with_invalid_mapping(
        self,
        protocol_short,
        benzene_solvent_system,
        toluene_solvent_system,
        mapping_benzonitrile_styrene,
    ):
        """Mapping whose components don't match the states should raise AssertionError."""
        with pytest.raises(AssertionError):
            _ = protocol_short.create(
                stateA=benzene_solvent_system,
                stateB=toluene_solvent_system,
                name="bad mapping",
                mapping=mapping_benzonitrile_styrene,
            )

    def test_error_with_multiple_mappings(
        self,
        protocol_short,
        benzene_vacuum_system,
        toluene_vacuum_system,
        mapping_benzene_toluene,
    ):
        """Passing a list of more than one mapping should raise ValueError."""
        with pytest.raises(ValueError):
            _ = protocol_short.create(
                stateA=benzene_vacuum_system,
                stateB=toluene_vacuum_system,
                name="multiple mappings",
                mapping=[mapping_benzene_toluene, mapping_benzene_toluene],
            )

    def test_fail_with_multiple_solvent_comps(
        self,
        protocol_short,
        benzene_solvent_system,
        toluene_double_solvent_system,
        mapping_benzene_toluene,
    ):
        """A state with more than one solvent component should raise AssertionError."""
        with pytest.raises(AssertionError):
            _ = protocol_short.create(
                stateA=benzene_solvent_system,
                stateB=toluene_double_solvent_system,
                name="double solvent",
                mapping=mapping_benzene_toluene,
            )

    def test_create_execute_gather(
        self,
        short_switching_settings_multiple_switches,
        benzene_vacuum_system,
        toluene_vacuum_system,
        mapping_benzene_toluene,
        tmpdir,
    ):
        """
        Run the switching protocol with multiple switches and gather results.
        Checks that the execution is successful and that the free energy estimate
        and its uncertainty are not NaN.
        """
        import numpy as np

        protocol = NonEquilibriumSwitchingProtocol(
            settings=short_switching_settings_multiple_switches
        )
        dag = protocol.create(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            name="Multiple switches vacuum",
            mapping=mapping_benzene_toluene,
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

        for dag_result in results:
            assert (
                len(dag_result.protocol_unit_failures) == 0
            ), "Unit failure in protocol dag result."

        protocolresult = protocol.gather(results)
        fe_estimate = protocolresult.get_estimate()
        fe_error = protocolresult.get_uncertainty()
        assert not np.isnan(fe_estimate.magnitude), "Free energy estimate is NaN."
        assert not np.isnan(fe_error.magnitude), "Free energy uncertainty is NaN."


def test_settings_round_trip():
    """Settings must survive a JSON round-trip unchanged."""
    neq_settings = NonEquilibriumSwitchingProtocol.default_settings()
    neq_json = json.dumps(neq_settings.model_dump(), cls=JSON_HANDLER.encoder)
    neq_settings_2 = NonEquilibriumSwitchingSettings.model_validate(
        json.loads(neq_json, cls=JSON_HANDLER.decoder)
    )
    assert neq_settings == neq_settings_2
