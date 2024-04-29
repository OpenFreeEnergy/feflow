"""
Settings objects for the different integrators used in the protocols.

The reasoning is to have a base class that contains the settings that
would be shared between different integrators, and subclasses of it
for the specific integrator settings.
"""

try:
    from pydantic.v1 import validator
except ImportError:
    from pydantic import validator  # type: ignore[assignment]

from openff.units import unit
from openff.models.types import FloatQuantity
from gufe.settings import SettingsBaseModel


class PeriodicNonequilibriumIntegratorSettings(SettingsBaseModel):
    """Settings for the PeriodicNonequilibriumIntegrator"""

    class Config:
        arbitrary_types_allowed = True

    timestep: FloatQuantity["femtosecond"] = 4 * unit.femtoseconds
    """Size of the simulation timestep. Default 4 fs."""
    splitting: str = "V R H O R V"
    """Operator splitting"""
    equilibrium_steps: int = 12500
    """Number of steps for the equilibrium parts of the cycle. Default 250000"""
    nonequilibrium_steps: int = 12500
    """Number of steps for the non-equilibrium parts of the cycle. Default 250000"""
    barostat_frequency: FloatQuantity['timestep'] = 25 * unit.timestep  # todo: IntQuantity
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
    @validator("timestep")
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = f"timestep must be positive, received {v}."
            raise ValueError(errmsg)
        return v

    # TODO: This validator is used in other settings, better create a new Type
    @validator("timestep")
    def is_time(cls, v):
        # these are time units, not simulation steps
        if not v.is_compatible_with(unit.picosecond):
            raise ValueError("timestep must be in time units " "(i.e. picoseconds)")
        return v

    # TODO: This validator is used in other settings, better create a new Type
    @validator("equilibrium_steps", "nonequilibrium_steps")
    def must_be_positive_or_zero(cls, v):
        if v < 0:
            errmsg = (
                "langevin_collision_rate, and n_restart_attempts must be"
                f" zero or positive values, got {v}."
            )
            raise ValueError(errmsg)
        return v
