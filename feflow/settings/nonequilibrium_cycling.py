"""
Settings objects for the different protocols using gufe objects.

This module implements the objects that will be needed to run relative binding free
energy calculations using perses.
"""

from typing import Optional

from feflow.settings import PeriodicNonequilibriumIntegratorSettings, OpenFFPartialChargeSettings

from gufe.settings import Settings
from pydantic import root_validator
from openfe.protocols.openmm_utils.omm_settings import (
    OpenMMSolvationSettings,
    OpenMMEngineSettings,
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
    class Config:
        arbitrary_types_allowed = True

    forcefield_cache: Optional[str] = (
        "db.json"  # TODO: Remove once it has been integrated with openfe settings
    )

    # Solvation settings
    solvation_settings: OpenMMSolvationSettings
    partial_charge_settings: OpenFFPartialChargeSettings
    """Settings for assigning partial charges to small molecules."""

    # Lambda settings
    lambda_functions = DEFAULT_ALCHEMICAL_FUNCTIONS

    # alchemical settings
    alchemical_settings: AlchemicalSettings = AlchemicalSettings(softcore_LJ="gapsys")

    # integrator settings
    integrator_settings: PeriodicNonequilibriumIntegratorSettings

    # platform and serialization
    engine_settings: OpenMMEngineSettings  # This defines platform
    traj_save_frequency: int = 2000
    work_save_frequency: int = 500
    atom_selection_expression: str = "not water"  # TODO: no longer used

    num_cycles: int = 100  # Number of cycles to run

    @root_validator
    def save_frequencies_consistency(cls, values):
        """Checks trajectory save frequency is a multiple of work save frequency, for convenience"""
        if values.get("traj_save_frequency") % values.get("work_save_frequency") != 0:
            raise ValueError(
                "Work save frequency must be a divisor of trajectory save frequency. "
                "Please specify consistent values for trajectory and work save settings"
            )
        # TODO: Add check for eq and neq steps and save frequencies
        return values
