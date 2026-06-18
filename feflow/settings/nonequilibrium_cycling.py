"""
Settings objects for the different protocols using gufe objects.
"""

from typing import Optional

from feflow.settings import (
    PeriodicNonequilibriumIntegratorSettings,
    OpenFFPartialChargeSettings,
)

from gufe.settings import Settings, OpenMMSystemGeneratorFFSettings
from pydantic import ConfigDict, model_validator
from openfe.protocols.openmm_utils.omm_settings import (
    OpenMMSolvationSettings,
    OpenMMEngineSettings,
    ThermoSettings,
)
from openfe.protocols.openmm_rfe.equil_rfe_settings import AlchemicalSettings

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


class NonEquilibriumCyclingSettings(Settings):
    """
    Settings for the NEQ cycling protocol.

    Attributes
    ----------
    ligand_input : str
        The path to the ligand input file.
    ligand_index : int
        The index of the ligand in the ligand input file.
    solvent_padding : float
        The amount of padding to add to the ligand in nanometers.
    forcefield : ForceFieldSettings
        The force field settings to use.
    alchemical : AlchemicalSettings
        The alchemical settings to use.
    """

    # TODO: Add type hints
    model_config = ConfigDict(arbitrary_types_allowed=True)

    forcefield_cache: Optional[str] = (
        "db.json"  # TODO: Remove once it has been integrated with openfe settings
    )
    forcefield_settings: OpenMMSystemGeneratorFFSettings
    # Solvation settings
    solvation_settings: OpenMMSolvationSettings
    partial_charge_settings: OpenFFPartialChargeSettings
    """Settings for assigning partial charges to small molecules."""

    # Lambda settings
    lambda_functions: dict[str, str] = DEFAULT_ALCHEMICAL_FUNCTIONS

    # alchemical settings
    alchemical_settings: AlchemicalSettings = AlchemicalSettings(softcore_LJ="gapsys")

    # integrator settings
    integrator_settings: PeriodicNonequilibriumIntegratorSettings

    # Thermodynamic settings
    thermo_settings: ThermoSettings

    # platform and serialization
    engine_settings: OpenMMEngineSettings  # This defines platform
    # TODO: Need to do validation checking on these values related to IntegratorSettings eq/neq steps
    traj_save_frequency: int = 2000
    work_save_frequency: int = 500
    atom_selection_expression: str = "not water"  # TODO: no longer used

    num_cycles: int = 100  # Number of cycles to run

    setup_minimize: bool = (
        True  # If True, minimize the system in the SetupUnit; we don't want to do this on platforms like Folding@Home
    )

    # Debugging settings
    store_minimized_pdb: bool = True
    """Setting for storing pdb right after minimization (right before neq cycle)"""

    @model_validator(mode="after")
    def save_frequencies_consistency(self):
        """Checks trajectory save frequency is a multiple of work save frequency, for convenience"""
        if self.traj_save_frequency % self.work_save_frequency != 0:
            raise ValueError(
                "Work save frequency must be a divisor of trajectory save frequency. "
                "Please specify consistent values for trajectory and work save settings"
            )
        # TODO: Add check for eq and neq steps and save frequencies
        return self

    @model_validator(mode="after")
    def store_minimized_pdb_requires_setup_minimize(self):
        """Storing the minimized PDB requires minimization to be enabled."""
        if self.store_minimized_pdb and not self.setup_minimize:
            raise ValueError(
                "`store_minimized_pdb` requires `setup_minimize` to be True, "
                "since there is no minimized structure to store when minimization "
                "is disabled."
            )
        return self
