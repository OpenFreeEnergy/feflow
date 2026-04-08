# Nonequilibrium switching protocol using AlchemicalNonequilibriumLangevinIntegrator.
# Reuses SetupUnit from nonequilibrium_cycling for hybrid topology construction.

import datetime
import logging
import re
import time
from typing import Optional, Any
from collections.abc import Iterable

from gufe.chemicalsystem import ChemicalSystem
from gufe.mapping import ComponentMapping
from gufe.protocols import (
    Protocol,
    ProtocolUnit,
    ProtocolResult,
    ProtocolDAGResult,
)

from openfe.protocols.openmm_utils.omm_compute import get_openmm_platform
from openff.units import unit
from openff.units.openmm import to_openmm

from ..settings import NonEquilibriumSwitchingSettings
from ..settings.nonequilibrium_switching import SnapshotSettings
from ..settings.small_molecules import OpenFFPartialChargeSettings
from ..utils.data import deserialize, serialize
from .nonequilibrium_cycling import SetupUnit  # reuse hybrid topology setup

logger = logging.getLogger(__name__)


def _reversed_lambda_functions(lambda_functions: dict[str, str]) -> dict[str, str]:
    """
    Derive alchemical functions for the reverse switch (lambda 1->0) by replacing
    the standalone 'lambda' variable with '(1.0 - lambda)' in all expressions.
    Uses a word-boundary regex to avoid partial matches.
    """
    return {
        k: re.sub(r"\blambda\b", "(1.0 - lambda)", v)
        for k, v in lambda_functions.items()
    }


def _load_snapshot(snapshot_settings: SnapshotSettings, index: int):
    """
    Load positions and box vectors for replicate *index* from a trajectory
    using MDAnalysis.

    Parameters
    ----------
    snapshot_settings : SnapshotSettings
    index : int
        Replicate index; selects frame ``index * stride`` from the trajectory.

    Returns
    -------
    positions_nm : np.ndarray, shape (n_atoms, 3)
        Positions in nanometres.
    box_vectors_nm : np.ndarray, shape (3, 3) or None
        Triclinic box vectors in nanometres, or None for vacuum systems.
    """
    import MDAnalysis as mda

    u = mda.Universe(snapshot_settings.topology_file, snapshot_settings.trajectory_file)
    frame_idx = index * snapshot_settings.stride
    u.trajectory[frame_idx]  # We need to place ourselves in the specified frame

    positions_nm = u.atoms.positions * 0.1  # Angstrom -> nm

    ts = u.trajectory.ts
    if ts.dimensions is not None:
        from MDAnalysis.lib.mdamath import triclinic_vectors

        box_vectors_nm = triclinic_vectors(ts.dimensions) * 0.1  # Angstrom -> nm
    else:
        box_vectors_nm = None

    return positions_nm, box_vectors_nm


class _BaseEquilibrationUnit(ProtocolUnit):
    """
    Produces ``num_switches`` equilibrated starting snapshots for one lambda
    endpoint.  Subclasses set ``_snapshot_settings_key`` and ``_endpoint``.

    Two modes:
    - **Internal equilibration**: runs a single continuous Langevin trajectory
      for ``equilibrium_steps`` total steps, saving ``num_switches`` snapshots
      at uniform intervals (every ``equilibrium_steps // num_switches`` steps).
      Raises if ``equilibrium_steps < num_switches``.
    - **XTC trajectory**: loads ``num_switches`` frames from a pre-equilibrated
      trajectory via MDAnalysis (frame ``i * stride``).  Raises if the
      trajectory does not contain enough frames.

    Outputs
    -------
    snap_states : list[pathlib.Path]
        Serialized OpenMM State XML files, one per switch replicate.
    timing_info : dict
    log : pathlib.Path
    """

    _snapshot_settings_key: str = ""  # "lambda0_snapshots" or "lambda1_snapshots"
    _endpoint: str = ""  # "lambda0" or "lambda1"

    def _execute(self, ctx, *, protocol, setup, **inputs):
        import openmm
        import openmm.unit as openmm_unit

        settings: NonEquilibriumSwitchingSettings = protocol.settings
        int_settings = settings.integrator_settings

        temperature = to_openmm(settings.thermo_settings.temperature)
        timestep = to_openmm(int_settings.timestep)
        collision_rate = to_openmm(int_settings.collision_rate)
        eq_steps = int_settings.equilibrium_steps
        num_switches = settings.num_switches

        file_logger = logging.getLogger(f"neq-eq-{self._endpoint}")
        log_path = ctx.shared / f"feflow-eq-{self._endpoint}-{self.name}.log"
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        file_logger.addHandler(file_handler)

        system = deserialize(setup.outputs["system"])
        initial_state = deserialize(setup.outputs["state"])
        platform = get_openmm_platform(settings.engine_settings.compute_platform)
        timing_info = {}

        snapshot_settings: Optional[SnapshotSettings] = getattr(
            settings, self._snapshot_settings_key
        )

        snap_states = []

        if snapshot_settings is not None:
            # Load num_switches frames from a pre-equilibrated XTC trajectory
            import MDAnalysis as mda

            u = mda.Universe(
                snapshot_settings.topology_file, snapshot_settings.trajectory_file
            )
            available = len(u.trajectory) // snapshot_settings.stride
            if available < num_switches:
                raise ValueError(
                    f"{self._endpoint} trajectory has only {len(u.trajectory)} frames "
                    f"(stride={snapshot_settings.stride} → {available} usable), "
                    f"but num_switches={num_switches}."
                )

            file_logger.info(
                f"{self.name}: loading {num_switches} snapshots from "
                f"{snapshot_settings.trajectory_file} (stride={snapshot_settings.stride})"
            )
            integrator = openmm.LangevinMiddleIntegrator(
                temperature, collision_rate, timestep
            )
            ctx_snap = openmm.Context(system, integrator, platform)
            ctx_snap.setState(initial_state)

            t0 = time.perf_counter()
            for i in range(num_switches):
                positions_nm, box_nm = _load_snapshot(snapshot_settings, i)
                ctx_snap.setPositions(positions_nm * openmm_unit.nanometers)
                if box_nm is not None:
                    ctx_snap.setPeriodicBoxVectors(*box_nm * openmm_unit.nanometers)
                ctx_snap.setVelocitiesToTemperature(temperature)
                snap_states.append(
                    ctx_snap.getState(
                        getPositions=True,
                        getVelocities=True,
                        getEnergy=True,
                        getForces=True,
                        enforcePeriodicBox=True,
                    )
                )
            del ctx_snap, integrator
            timing_info["snapshot_load_time_in_s"] = datetime.timedelta(
                seconds=time.perf_counter() - t0
            ).total_seconds()
            file_logger.info(
                f"{self.name}: loaded {num_switches} snapshots "
                f"({timing_info['snapshot_load_time_in_s']:.1f} s)"
            )

        else:
            # Internal equilibration: single continuous trajectory of eq_steps
            # total steps; snapshots taken at uniform intervals.
            if eq_steps < num_switches:
                raise ValueError(
                    f"equilibrium_steps ({eq_steps}) must be >= num_switches "
                    f"({num_switches}) to produce uniformly-spaced snapshots."
                )

            save_interval = eq_steps // num_switches
            file_logger.info(
                f"{self.name}: running {eq_steps} equilibration steps, "
                f"saving {num_switches} snapshots every {save_interval} steps"
            )
            eq_integrator = openmm.LangevinMiddleIntegrator(
                temperature, collision_rate, timestep
            )
            eq_ctx = openmm.Context(system, eq_integrator, platform)
            eq_ctx.setState(initial_state)
            eq_ctx.setVelocitiesToTemperature(temperature)

            t0 = time.perf_counter()
            for i in range(num_switches):
                # We run eq simulations and save snapshots where each switch should start
                eq_integrator.step(save_interval)
                snap_states.append(
                    eq_ctx.getState(
                        getPositions=True,
                        getVelocities=True,
                        getEnergy=True,
                        getForces=True,
                        enforcePeriodicBox=True,
                    )
                )
            del eq_ctx, eq_integrator
            timing_info["eq_time_in_s"] = datetime.timedelta(
                seconds=time.perf_counter() - t0
            ).total_seconds()
            file_logger.info(
                f"{self.name}: equilibration done ({timing_info['eq_time_in_s']:.1f} s)"
            )

        # Serialize each snapshot to its own XML file
        snap_paths = []
        for i, state in enumerate(snap_states):
            path = ctx.shared / f"{self._endpoint}_snapshot_state_{self.name}_{i}.xml"
            serialize(state, path)
            snap_paths.append(path)

        return {
            "snap_states": snap_paths,
            "timing_info": timing_info,
            "log": log_path,
        }


class Lambda0EquilibrationUnit(_BaseEquilibrationUnit):
    """Equilibration / snapshot loading at lambda=0 (starting point for forward switches)."""

    _snapshot_settings_key = "lambda0_snapshots"
    _endpoint = "lambda0"


class Lambda1EquilibrationUnit(_BaseEquilibrationUnit):
    """Equilibration / snapshot loading at lambda=1 (starting point for reverse switches)."""

    _snapshot_settings_key = "lambda1_snapshots"
    _endpoint = "lambda1"


class _BaseSwitchingUnit(ProtocolUnit):
    """
    Shared machinery for ForwardSwitchingUnit and ReverseSwitchingUnit.
    Subclasses provide ``_direction`` and ``_get_lambda_functions``.
    """

    @staticmethod
    def extract_positions(context, initial_atom_indices, final_atom_indices):

        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        return (
            positions[list(initial_atom_indices), :],
            positions[list(final_atom_indices), :],
        )

    # --- to be overridden by subclasses ---
    _direction: str = ""  # "forward" or "reverse"

    def _get_lambda_functions(self, settings: NonEquilibriumSwitchingSettings) -> dict:
        raise NotImplementedError

    # --------------------------------------------------------------

    def _execute(self, ctx, *, protocol, setup, equilibration, index, **inputs):
        """
        Execute one NEQ switch starting from the pre-equilibrated snapshot
        produced by an upstream EquilibrationUnit.

        Parameters
        ----------
        ctx : gufe.protocols.protocolunit.Context
        protocol : NonEquilibriumSwitchingProtocol
        setup : SetupUnit result with hybrid system and atom indices.
        equilibration : EquilibrationUnit result with the starting snapshot.
        index : int
            Replicate index; used for output file naming.
        """
        import numpy as np
        import openmm
        from openmmtools.integrators import AlchemicalNonequilibriumLangevinIntegrator

        # Logging
        file_logger = logging.getLogger(f"neq-switching-{self._direction}")
        log_path = ctx.shared / f"feflow-neq-{self._direction}-{self.name}.log"
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        file_logger.addHandler(file_handler)

        settings: NonEquilibriumSwitchingSettings = protocol.settings
        int_settings = settings.integrator_settings

        temperature = to_openmm(settings.thermo_settings.temperature)
        timestep = to_openmm(int_settings.timestep)
        collision_rate = to_openmm(int_settings.collision_rate)
        neq_steps = int_settings.nonequilibrium_steps
        work_save_freq = settings.work_save_frequency
        traj_save_freq = settings.traj_save_frequency
        coarse_steps = neq_steps // work_save_freq
        traj_coarse_freq = traj_save_freq // work_save_freq

        system = deserialize(setup.outputs["system"])
        initial_atom_indices = setup.outputs["initial_atom_indices"]
        final_atom_indices = setup.outputs["final_atom_indices"]

        snap_state = deserialize(equilibration.outputs["snap_states"][index])

        platform = get_openmm_platform(settings.engine_settings.compute_platform)
        timing_info = {}

        # ------------------------------------------------------------------
        # NEQ switch
        # ------------------------------------------------------------------
        lambda_functions = self._get_lambda_functions(settings)

        neq_integrator = AlchemicalNonequilibriumLangevinIntegrator(
            alchemical_functions=lambda_functions,
            splitting=int_settings.splitting,
            temperature=temperature,
            collision_rate=collision_rate,
            timestep=timestep,
            nsteps_neq=neq_steps,
        )
        neq_ctx = openmm.Context(system, neq_integrator, platform)
        neq_ctx.setState(snap_state)

        # Adding minimization in the base switching unit. Helped to avoid NaNs in cases.
        t_min0 = time.perf_counter()
        openmm.LocalEnergyMinimizer.minimize(neq_ctx)
        timing_info["minimization_time_in_s"] = (
            time.perf_counter() - t_min0
        )
        file_logger.info(
            f"{self.name}: minimized starting snapshot "
            f"({timing_info['minimization_time_in_s']:.1f} s)"
        )

        works = [neq_integrator.get_protocol_work(dimensionless=True)]
        initial_traj, final_traj = [], []

        file_logger.info(
            f"{self.name}: {self._direction} NEQ switch ({neq_steps} steps)"
        )
        t0 = time.perf_counter()
        try:
            for step in range(coarse_steps):
                neq_integrator.step(work_save_freq)
                works.append(neq_integrator.get_protocol_work(dimensionless=True))
                if step % traj_coarse_freq == 0:
                    i_pos, f_pos = self.extract_positions(
                        neq_ctx, initial_atom_indices, final_atom_indices
                    )
                    initial_traj.append(i_pos)
                    final_traj.append(f_pos)
            # Always capture the final frame
            i_pos, f_pos = self.extract_positions(
                neq_ctx, initial_atom_indices, final_atom_indices
            )
            initial_traj.append(i_pos)
            final_traj.append(f_pos)
        finally:
            del neq_ctx, neq_integrator

        neq_elapsed = datetime.timedelta(seconds=time.perf_counter() - t0)
        timing_info[f"neq_{self._direction}_time_in_s"] = neq_elapsed.total_seconds()
        file_logger.info(f"{self.name}: NEQ switch time: {neq_elapsed}")

        # ------------------------------------------------------------------
        # Serialize outputs
        # ------------------------------------------------------------------
        work_path = ctx.shared / f"{self._direction}_{self.name}.npy"
        initial_traj_path = ctx.shared / f"{self._direction}_initial_{self.name}.npy"
        final_traj_path = ctx.shared / f"{self._direction}_final_{self.name}.npy"

        with open(work_path, "wb") as f:
            np.save(f, works)
        with open(initial_traj_path, "wb") as f:
            np.save(f, np.array(initial_traj))
        with open(final_traj_path, "wb") as f:
            np.save(f, np.array(final_traj))

        return {
            "work": work_path,
            "initial_traj": initial_traj_path,
            "final_traj": final_traj_path,
            "timing_info": timing_info,
            "log": log_path,
        }


class ForwardSwitchingUnit(_BaseSwitchingUnit):
    """Runs one forward NEQ switch (lambda 0->1)."""

    _direction = "forward"

    def _get_lambda_functions(self, settings):
        return settings.lambda_functions


class ReverseSwitchingUnit(_BaseSwitchingUnit):
    """Runs one reverse NEQ switch (lambda 1->0)."""

    _direction = "reverse"

    def _get_lambda_functions(self, settings):
        return _reversed_lambda_functions(settings.lambda_functions)


class ResultUnit(ProtocolUnit):
    """Collects per-switch work arrays from all ForwardSwitchingUnits and ReverseSwitchingUnits."""

    @staticmethod
    def _execute(ctx, *, forward_switches, reverse_switches, **inputs):
        import numpy as np

        forward_works, reverse_works = [], []
        for unit in forward_switches:
            w = np.load(unit.outputs["work"])
            forward_works.append(w - w[0])
        for unit in reverse_switches:
            w = np.load(unit.outputs["work"])
            reverse_works.append(w - w[0])

        return {
            "forward_work": forward_works,
            "reverse_work": reverse_works,
            "forward_work_paths": [u.outputs["work"] for u in forward_switches],
            "reverse_work_paths": [u.outputs["work"] for u in reverse_switches],
        }


class NonEquilibriumSwitchingProtocolResult(ProtocolResult):
    """
    Collects results and computes free energy estimates via BAR from the
    forward (lambda 0->1) and reverse (lambda 1->0) work values.
    """

    def get_estimate(self):
        """Free energy estimate via BAR in kcal/mol."""
        import numpy as np
        import numpy.typing as npt
        import pymbar

        forward: npt.NDArray = np.array([w[-1] for w in self.data["forward_work"]])
        reverse: npt.NDArray = np.array([w[-1] for w in self.data["reverse_work"]])
        bar_data = pymbar.bar(forward, reverse)
        return (
            bar_data["Delta_f"]
            * unit.k
            * self.data["temperature"]
            * unit.avogadro_constant
        ).to("kcal/mol")

    def get_uncertainty(self, n_bootstraps: int = 1000):
        """Uncertainty via bootstrapped BAR standard deviation in kcal/mol."""
        import numpy as np
        import numpy.typing as npt

        forward: npt.NDArray = np.array([w[-1] for w in self.data["forward_work"]])
        reverse: npt.NDArray = np.array([w[-1] for w in self.data["reverse_work"]])
        all_dgs = self._do_bootstrap(forward, reverse, n_bootstraps)
        return (
            np.std(all_dgs) * unit.k * self.data["temperature"] * unit.avogadro_constant
        ).to("kcal/mol")

    def get_rate_of_convergence(self): ...

    def _do_bootstrap(self, forward, reverse, n_bootstraps: int = 1000):
        import numpy as np
        import pymbar

        assert len(forward) == len(
            reverse
        ), "Forward and reverse work arrays must have the same length."
        n = len(forward)
        all_dgs = np.zeros(n_bootstraps)
        for i in range(n_bootstraps):
            idx = np.random.choice(np.arange(n), size=n, replace=True)
            all_dgs[i] = pymbar.bar(forward[idx], reverse[idx])["Delta_f"]
        return all_dgs


class NonEquilibriumSwitchingProtocol(Protocol):
    """
    RBFE protocol using non-equilibrium switching with the
    AlchemicalNonequilibriumLangevinIntegrator from openmmtools.

    For each of ``num_switches`` replicates the protocol creates:
    - one ForwardSwitchingUnit  (lambda 0->1)
    - one ReverseSwitchingUnit  (lambda 1->0)

    Both sets of units depend only on the shared SetupUnit and can therefore
    run in parallel. Free energy is estimated via BAR over all work values.

    Starting snapshots are either generated by internal equilibration
    (``integrator_settings.equilibrium_steps > 0``) or loaded from
    pre-equilibrated trajectories via ``lambda0_snapshots`` / ``lambda1_snapshots``.
    """

    _settings_cls = NonEquilibriumSwitchingSettings
    result_cls = NonEquilibriumSwitchingProtocolResult

    @classmethod
    def _default_settings(cls):
        from feflow.settings import (
            NonEquilibriumSwitchingSettings,
            AlchemicalNonequilibriumIntegratorSettings,
        )
        from gufe.settings import OpenMMSystemGeneratorFFSettings
        from openfe.protocols.openmm_utils.omm_settings import (
            OpenMMSolvationSettings,
            OpenMMEngineSettings,
            ThermoSettings,
        )
        from openfe.protocols.openmm_rfe.equil_rfe_settings import AlchemicalSettings

        return NonEquilibriumSwitchingSettings(
            forcefield_settings=OpenMMSystemGeneratorFFSettings(),
            thermo_settings=ThermoSettings(
                temperature=300 * unit.kelvin, pressure=1 * unit.bar
            ),
            solvation_settings=OpenMMSolvationSettings(),
            partial_charge_settings=OpenFFPartialChargeSettings(),
            alchemical_settings=AlchemicalSettings(softcore_LJ="gapsys"),
            integrator_settings=AlchemicalNonequilibriumIntegratorSettings(),
            engine_settings=OpenMMEngineSettings(),
        )

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[ComponentMapping | list[ComponentMapping]],
        extends: Optional[ProtocolDAGResult] = None,
    ) -> list[ProtocolUnit]:
        if isinstance(mapping, list):
            if len(mapping) != 1:
                raise ValueError(
                    "Exactly one mapping must be provided for this protocol."
                )
            mapping = mapping[0]
        self.validate(stateA=stateA, stateB=stateB, mapping=mapping, extends=extends)

        num_switches = self.settings.num_switches

        setup = SetupUnit(
            protocol=self,
            state_a=stateA,
            state_b=stateB,
            mapping=mapping,
            name="setup",
        )

        # One equilibration unit per lambda endpoint; each produces num_switches
        # snapshots and depends only on setup.
        lambda0_eq = Lambda0EquilibrationUnit(
            protocol=self, setup=setup, name="eq_lambda0"
        )
        lambda1_eq = Lambda1EquilibrationUnit(
            protocol=self, setup=setup, name="eq_lambda1"
        )

        # Switching units depend on setup (system/indices) and the appropriate
        # equilibration unit (starting snapshot by index).
        forward_switches = [
            ForwardSwitchingUnit(
                protocol=self,
                setup=setup,
                equilibration=lambda0_eq,
                index=i,
                name=f"forward_{i}",
            )
            for i in range(num_switches)
        ]
        reverse_switches = [
            ReverseSwitchingUnit(
                protocol=self,
                setup=setup,
                equilibration=lambda1_eq,
                index=i,
                name=f"reverse_{i}",
            )
            for i in range(num_switches)
        ]

        end = ResultUnit(
            name="result",
            forward_switches=forward_switches,
            reverse_switches=reverse_switches,
        )

        return [setup, lambda0_eq, lambda1_eq, *forward_switches, *reverse_switches, end]

    @staticmethod
    def _check_mappings_consistency(mapping, chemical_system_a, chemical_system_b):
        mapping_comp_a = mapping.componentA
        mapping_comp_b = mapping.componentB
        chem_sys_a_keys = [c.key for _, c in chemical_system_a.components.items()]
        chem_sys_b_keys = [c.key for _, c in chemical_system_b.components.items()]
        assert (
            mapping_comp_a.key in chem_sys_a_keys
        ), "Component A in mapping not found in chemical system A."
        assert (
            mapping_comp_b.key in chem_sys_b_keys
        ), "Component B in mapping not found in chemical system B."

    def _validate(
        self,
        *,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: ComponentMapping | list[ComponentMapping] | None,
        extends: Optional[ProtocolDAGResult] = None,
    ):
        from gufe import SolventComponent

        if mapping is None:
            raise ValueError("`mapping` is required for this Protocol")
        if extends:
            raise NotImplementedError("Can't extend simulations yet")

        self._check_mappings_consistency(
            mapping=mapping, chemical_system_a=stateA, chemical_system_b=stateB
        )

        state_a_solv = stateA.get_components_of_type(SolventComponent)
        state_b_solv = stateB.get_components_of_type(SolventComponent)
        assert (
            len(state_a_solv) <= 1
        ), f"State A has {len(state_a_solv)} solvent components. Only 0 or 1 allowed."
        assert (
            len(state_b_solv) <= 1
        ), f"State B has {len(state_b_solv)} solvent components. Only 0 or 1 allowed."

    def _gather(
        self, protocol_dag_results: Iterable[ProtocolDAGResult]
    ) -> dict[str, Any]:
        from collections import defaultdict

        outputs = defaultdict(list)
        for pdr in protocol_dag_results:
            for pur in pdr.protocol_unit_results:
                if pur.name == "result":
                    outputs["forward_work"].extend(pur.outputs["forward_work"])
                    outputs["reverse_work"].extend(pur.outputs["reverse_work"])

        outputs["temperature"] = self.settings.thermo_settings.temperature
        return outputs
