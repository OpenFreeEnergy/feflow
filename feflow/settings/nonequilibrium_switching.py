"""
Settings objects for the nonequilibrium switching protocol.
"""

from typing import Optional

from feflow.settings import (
    AlchemicalNonequilibriumIntegratorSettings,
    OpenFFPartialChargeSettings,
)

from gufe.settings import Settings, OpenMMSystemGeneratorFFSettings, SettingsBaseModel
from openfe.protocols.openmm_utils.omm_settings import (
    OpenMMSolvationSettings,
    OpenMMEngineSettings,
    ThermoSettings,
)
from openfe.protocols.openmm_rfe.equil_rfe_settings import AlchemicalSettings
from pydantic import ConfigDict, model_validator


# Default settings for the lambda functions
x = "lambda"
DEFAULT_ALCHEMICAL_FUNCTIONS = {
    "lambda_sterics_core": x,
    "lambda_electrostatics_core": x,
    "lambda_sterics_insert": f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
    "lambda_sterics_delete": f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
    "lambda_electrostatics_insert": f"select(step({x} - 0.5), 2.0 * ({x} - 0.5), 0.0)",
    "lambda_electrostatics_delete": f"select(step({x} - 0.5), 1.0, 2.0 * {x})",
    "lambda_bonds": x,
    "lambda_angles": x,
    "lambda_torsions": x,
}


class SnapshotSettings(SettingsBaseModel):
    """
    Settings for loading pre-equilibrated snapshots from a trajectory file via
    MDAnalysis, instead of running internal equilibration.

    The trajectory must have been generated from the **hybrid topology** produced
    by the SetupUnit (i.e. the same system that will be used for NEQ switching).
    Replicate ``i`` uses frame ``i * stride`` from the trajectory.
    """

    topology_file: str
    """Path to a topology file (PDB, GRO, …) compatible with the hybrid topology."""
    trajectory_file: str
    """Path to a trajectory file (XTC, DCD, …) of equilibrated configurations."""
    stride: int = 1
    """
    Frame stride for snapshot selection.
    Replicate i loads frame i * stride. Default 1 (consecutive frames).
    """


class NonEquilibriumSwitchingSettings(Settings):
    """
    Settings for the NEQ switching protocol.

    The protocol drives the hybrid system from lambda=0 to lambda=1 (forward
    switches) and from lambda=1 to lambda=0 (reverse switches) using the
    AlchemicalNonequilibriumLangevinIntegrator from openmmtools. Free energy
    estimates are obtained via BAR over the replicate work values.

    Starting snapshots for each direction can either be generated internally
    via ``integrator_settings.equilibrium_steps`` (set ``lambda0_snapshots`` /
    ``lambda1_snapshots`` to ``None``) or loaded from an existing trajectory with
    MDAnalysis. When snapshot settings are provided, ``equilibrium_steps`` must be
    set to 0 to avoid ambiguity.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    forcefield_cache: Optional[str] = "db.json"

    forcefield_settings: OpenMMSystemGeneratorFFSettings
    solvation_settings: OpenMMSolvationSettings
    partial_charge_settings: OpenFFPartialChargeSettings
    thermo_settings: ThermoSettings

    # Lambda schedule
    lambda_functions: dict[str, str] = DEFAULT_ALCHEMICAL_FUNCTIONS

    # Alchemical settings (softcore potentials, charge corrections, …)
    alchemical_settings: AlchemicalSettings = AlchemicalSettings(softcore_LJ="gapsys")

    # Integrator (AlchemicalNonequilibriumLangevinIntegrator)
    integrator_settings: AlchemicalNonequilibriumIntegratorSettings

    # Platform
    engine_settings: OpenMMEngineSettings

    # Optional pre-equilibrated snapshot sources
    lambda0_snapshots: Optional[SnapshotSettings] = None
    """
    If provided, forward switching replicates load their starting snapshot at
    lambda=0 from this trajectory instead of running internal equilibration.
    Requires ``integrator_settings.equilibrium_steps == 0``.
    """
    lambda1_snapshots: Optional[SnapshotSettings] = None
    """
    If provided, reverse switching replicates load their starting snapshot at
    lambda=1 from this trajectory instead of running internal equilibration.
    Requires ``integrator_settings.equilibrium_steps == 0``.
    """

    work_save_frequency: Optional[int] = None
    """
    How often (in NEQ steps) to record the protocol work.
    Defaults to nonequilibrium_steps // 50, giving ~50 work samples per switch.
    """
    traj_save_frequency: Optional[int] = None
    """
    How often (in NEQ steps) to save trajectory frames during switching.
    Defaults to 5 * work_save_frequency (~10 frames per switch).
    Must be a multiple of work_save_frequency.
    """

    num_switches: int = 100
    """
    Number of independent NEQ switch trajectories to run per direction.
    The protocol creates this many forward (lambda 0->1) switches and this many
    reverse (lambda 1->0) switches, for a total of 2 * num_switches trajectory
    runs. Each switch produces one work value used in the BAR free energy estimate.
    """

    # Debugging settings
    store_minimized_pdb: bool = True
    """Setting for storing pdb right after minimization (right before neq cycle)"""

    @model_validator(mode="after")
    def set_and_validate_save_frequencies(self):
        """
        Derive save-frequency defaults from nonequilibrium_steps when not set,
        then check consistency.
        """
        neq_steps = (
            self.integrator_settings.nonequilibrium_steps
            if self.integrator_settings
            else 2500
        )

        if self.work_save_frequency is None:
            self.work_save_frequency = max(1, neq_steps // 50)
        if self.traj_save_frequency is None:
            self.traj_save_frequency = self.work_save_frequency * 5

        if self.traj_save_frequency % self.work_save_frequency != 0:
            raise ValueError(
                "traj_save_frequency must be a multiple of work_save_frequency. "
                "Please specify consistent values."
            )
        return self

    @model_validator(mode="after")
    def snapshots_require_no_internal_equilibration(self):
        """
        When snapshot settings are provided equilibrium_steps must be 0 —
        the snapshots are already equilibrated.
        """
        if self.integrator_settings is None:
            return self

        has_snapshots = self.lambda0_snapshots or self.lambda1_snapshots
        if has_snapshots and self.integrator_settings.equilibrium_steps != 0:
            raise ValueError(
                "When lambda0_snapshots or lambda1_snapshots are provided the "
                "snapshots are assumed to be pre-equilibrated. "
                "Set integrator_settings.equilibrium_steps = 0 to avoid "
                "running redundant equilibration on top of them."
            )
        return self

    @model_validator(mode="after")
    def snapshot_trajectories_have_enough_frames(self):
        """
        When both lambda0_snapshots and lambda1_snapshots are provided, check
        that each trajectory contains enough frames to cover all replicates
        (i.e. at least num_switches * stride frames), and that both
        trajectories expose the same number of usable snapshots.
        """
        snap0 = self.lambda0_snapshots
        snap1 = self.lambda1_snapshots

        if snap0 is None or snap1 is None:
            return self

        num_switches = self.num_switches

        try:
            import MDAnalysis as mda

            n0 = len(
                mda.Universe(snap0.topology_file, snap0.trajectory_file).trajectory
            )
            n1 = len(
                mda.Universe(snap1.topology_file, snap1.trajectory_file).trajectory
            )
        except Exception as exc:
            raise ValueError(
                f"Could not read snapshot trajectories to validate frame counts: {exc}"
            ) from exc

        usable0 = n0 // snap0.stride
        usable1 = n1 // snap1.stride

        if usable0 < num_switches:
            raise ValueError(
                f"lambda0 trajectory has only {n0} frames (stride={snap0.stride} → "
                f"{usable0} usable), but num_switches={num_switches}."
            )
        if usable1 < num_switches:
            raise ValueError(
                f"lambda1 trajectory has only {n1} frames (stride={snap1.stride} → "
                f"{usable1} usable), but num_switches={num_switches}."
            )
        if usable0 != usable1:
            raise ValueError(
                f"lambda0 and lambda1 trajectories expose different numbers of "
                f"usable snapshots ({usable0} vs {usable1}). "
                "Provide trajectories of equal length or adjust the stride values."
            )
        return self
