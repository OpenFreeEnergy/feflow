"""
Settings objects for the different integrators used in the protocols.

The reasoning is to have a base class that contains the settings that
would be shared between different integrators, and subclasses of it
for the specific integrator settings.
"""

from typing import Annotated, TypeAlias

from openff.units import unit
from gufe.settings import SettingsBaseModel
from gufe.settings.typing import GufeQuantity, specify_quantity_units
from pydantic import field_validator, ConfigDict

FemtosecondQuantity: TypeAlias = Annotated[
    GufeQuantity, specify_quantity_units("femtoseconds")
]
TimestepQuantity: TypeAlias = Annotated[
    GufeQuantity, specify_quantity_units("timestep")
]


class PeriodicNonequilibriumIntegratorSettings(SettingsBaseModel):
    """Settings for the PeriodicNonequilibriumIntegrator"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestep: FemtosecondQuantity = 4 * unit.femtoseconds
    """Size of the simulation timestep. Default 4 fs."""
    splitting: str = "V R H O R V"
    """Operator splitting"""
    equilibrium_steps: int = 12500
    """Number of steps for the equilibrium parts of the cycle. Default 12500"""
    nonequilibrium_steps: int = 12500
    """Number of steps for the non-equilibrium parts of the cycle. Default 12500"""
    barostat_frequency: TimestepQuantity = 25 * unit.timestep
    """
    Frequency at which volume scaling changes should be attempted.
    Note: The barostat frequency is ignored for gas-phase simulations.
    Default 25 * unit.timestep.
    """
    remove_com: bool = False
    """
    Whether or not to remove the center of mass motion. Default False.
    """

    # TODO: This validator is used in other settings, better create a new Type
    @field_validator("timestep")
    @classmethod
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = f"timestep must be positive, received {v}."
            raise ValueError(errmsg)
        return v

    # TODO: This validator is used in other settings, better create a new Type
    @field_validator("timestep")
    @classmethod
    def is_time(cls, v):
        # these are time units, not simulation steps
        if not v.is_compatible_with(unit.picosecond):
            raise ValueError("timestep must be in time units " "(i.e. picoseconds)")
        return v

    # TODO: This validator is used in other settings, better create a new Type
    @field_validator("equilibrium_steps", "nonequilibrium_steps")
    @classmethod
    def must_be_positive_or_zero(cls, v):
        if v < 0:
            errmsg = (
                "langevin_collision_rate, and n_restart_attempts must be"
                f" zero or positive values, got {v}."
            )
            raise ValueError(errmsg)
        return v
