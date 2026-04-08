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
InversePicosecondQuantity: TypeAlias = Annotated[
    GufeQuantity, specify_quantity_units("1/picoseconds")
]


class BaseNonequilibriumIntegrator(SettingsBaseModel):
    """Base class for nonequilibrium integrator settings"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestep: FemtosecondQuantity = 4 * unit.femtoseconds
    """Size of the simulation timestep. Default 4 fs."""
    splitting: str = "V R H O R V"
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


class AlchemicalNonequilibriumIntegratorSettings(BaseNonequilibriumIntegrator):
    """Settings for the AlchemicalNonequilibriumLangevinIntegrator used for one-way NEQ switching"""

    timestep: FemtosecondQuantity = 4 * unit.femtoseconds
    """Size of the simulation timestep. Default 4 fs."""
    splitting: str = "V R H O R V"
    """Operator splitting for the Langevin integrator."""
    collision_rate: InversePicosecondQuantity = 1.0 / unit.picoseconds
    """Langevin collision rate (friction coefficient). Default 1/ps."""
    nonequilibrium_steps: int = 2500
    """Number of steps for the non-equilibrium switching (lambda 0->1 or 1->0). Default 2500 (10 ps at 4 fs)."""
    equilibrium_steps: int = 1000
    """Number of equilibration steps at the endpoint before each switch. Default 1000."""

    @field_validator("nonequilibrium_steps", "equilibrium_steps")
    @classmethod
    def must_be_positive_or_zero(cls, v):
        if v < 0:
            errmsg = f"nonequilibrium_steps and equilibrium_steps must be zero or positive, got {v}."
            raise ValueError(errmsg)
        return v

    @field_validator("collision_rate")
    @classmethod
    def collision_rate_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError(f"collision_rate must be positive, received {v}.")
        return v


class PeriodicNonequilibriumIntegratorSettings(BaseNonequilibriumIntegrator):
    """Settings for the PeriodicNonequilibriumIntegrator"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
