from pathlib import Path

import pymbar.utils
import pytest

from feflow.protocols import NonEquilibriumCyclingProtocol
from gufe.protocols.protocoldag import ProtocolDAGResult, execute_DAG
from gufe.protocols.protocolunit import ProtocolUnitResult
from gufe.protocols.protocolunit import ProtocolUnitFailure


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
            mapping={"ligand": mapping_benzene_toluene},
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
    def protocol_dag_broken(
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
            mapping={"ligand": broken_mapping},
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

    @pytest.mark.parametrize(
        "protocol",
        [
            "protocol_short",
            "protocol_short_multiple_cycles",
        ],
    )
    def test_pdr_extend(
        self,
        protocol,
        benzene_vacuum_system,
        toluene_vacuum_system,
        mapping_benzene_toluene,
        tmpdir,
        request,
    ):

        protocol = request.getfixturevalue(protocol)
        dag = protocol.create(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            name="Short vacuum transformation",
            mapping={"ligand": mapping_benzene_toluene},
        )

        with tmpdir.as_cwd():

            base_path = Path("original")

            shared = base_path / "shared"
            shared.mkdir(parents=True)

            scratch = base_path / "scratch"
            scratch.mkdir(parents=True)

            pdr: ProtocolDAGResult = execute_DAG(
                dag, shared_basedir=shared, scratch_basedir=scratch
            )

        setup = pdr.protocol_units[0]
        r_setup = pdr.protocol_unit_results[0]

        assert setup.inputs["extends_data"] == {}

        end_states = {}
        for simulation, r_simulation in zip(
            pdr.protocol_units[1:-1], pdr.protocol_unit_results[1:-1]
        ):
            assert isinstance(r_simulation.outputs["system"], str)
            assert isinstance(r_simulation.outputs["state"], str)
            assert isinstance(r_simulation.outputs["integrator"], str)

            end_states[simulation.name] = r_simulation.outputs["state"]

        dag = protocol.create(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            name="Short vacuum transformation, but extended",
            mapping={"ligand": mapping_benzene_toluene},
            extends=ProtocolDAGResult.from_dict(pdr.to_dict()),
        )

        with tmpdir.as_cwd():

            base_path = Path("extended")

            shared = base_path / "shared"
            shared.mkdir(parents=True)

            scratch = base_path / "scratch"
            scratch.mkdir(parents=True)
            pdr: ProtocolDAGResult = execute_DAG(
                dag, shared_basedir=shared, scratch_basedir=scratch
            )

        r_setup = pdr.protocol_unit_results[0]

        assert r_setup.inputs["extends_data"] != {}

        for replicate in range(protocol.settings.num_replicates):
            replicate = str(replicate)
            assert isinstance(r_setup.inputs["extends_data"]["systems"][replicate], str)
            assert isinstance(r_setup.inputs["extends_data"]["states"][replicate], str)
            assert isinstance(
                r_setup.inputs["extends_data"]["integrators"][replicate], str
            )

            assert (
                r_setup.inputs["extends_data"]["states"][replicate]
                == end_states[replicate]
            )

    def test_dag_execute_failure(self, protocol_dag_broken):
        protocol, dag, dagfailure = protocol_dag_broken

        assert not dagfailure.ok()
        assert isinstance(dagfailure, ProtocolDAGResult)

        failed_units = dagfailure.protocol_unit_failures

        assert len(failed_units) == 1
        assert isinstance(failed_units[0], ProtocolUnitFailure)

    def test_dag_execute_failure_raise_error(
        self,
        protocol_short,
        benzene_vacuum_system,
        toluene_vacuum_system,
        broken_mapping,
        tmpdir,
    ):
        """Executes a bad setup of a protocol DAG which has an incorrect mapping"""
        dag = protocol_short.create(
            stateA=benzene_vacuum_system,
            stateB=toluene_vacuum_system,
            name="a broken dummy run",
            mapping={"ligand": broken_mapping},
        )

        # tries to access an atom index that does not exist
        with tmpdir.as_cwd():
            shared = Path("shared")
            shared.mkdir()

            scratch = Path("scratch")
            scratch.mkdir()

            with pytest.raises(IndexError):
                execute_DAG(
                    dag,
                    raise_error=True,
                    shared_basedir=shared,
                    scratch_basedir=scratch,
                )

    @pytest.mark.gpu_ci
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
            mapping={"ligand": mapping_benzene_toluene},
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

    @pytest.mark.gpu_ci
    @pytest.mark.parametrize(
        "protocol",
        [
            "protocol_short_multiple_cycles",
            #'protocol_short_multiple_cycles_gpu'
        ],
    )
    def test_create_execute_gather_toluene_to_toluene(
        self, protocol, toluene_vacuum_system, mapping_toluene_toluene, tmpdir, request
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
        """
        import numpy as np

        protocol = request.getfixturevalue(protocol)

        dag = protocol.create(
            stateA=toluene_vacuum_system,
            stateB=toluene_vacuum_system,
            name="Toluene vacuum transformation",
            mapping={"ligand": mapping_toluene_toluene},
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
    # TODO: Potentially setup (not run) a protein-ligand system
