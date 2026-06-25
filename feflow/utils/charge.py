"""
Module involving all the necessary auxiliary utilities and functions for manipulating charges.
Such as assigning both formal and partial charges, or transforming solvent into ions
or vice versa for charge-changing alchemical transformations.
"""

import logging
import warnings
from gufe import LigandAtomMapping, SolventComponent
from openfe.protocols.openmm_utils.charge_generation import (
    assign_offmol_partial_charges,
)

# TODO: Importing from OpenFE for now, should we migrate them here?
assign_offmol_partial_charges = assign_offmol_partial_charges

logger = logging.getLogger(__name__)


# TODO: Re-evaluate if we want a more global utility function for this in the openfe "ecosystem"
# Vendored from openfe protocol method in https://github.com/OpenFreeEnergy/openfe/blob/75cb2e85a46514633ecfe33353dfa5e9dc22e729/src/openfe/protocols/openmm_rfe/hybridtop_protocols.py#L373
def validate_charge_difference(
    mapping: LigandAtomMapping,
    nonbonded_method: str,
    explicit_charge_correction: bool,
    solvent_component: SolventComponent | None,
) -> int:
    """
    Validates the net charge difference between the two states.

    Useful for uses in Hybrid Topology protocols where alchemical changes
    of 2 or more charge units are not supported, and/or not using PME
    when there is charge correction is not supported.

    Parameters
    ----------
    mapping : LigandAtomMapping
      Mapping object between transforming components.
    nonbonded_method : str
      The OpenMM nonbonded method used for the simulation.
    explicit_charge_correction : bool
      Whether to use an explicit charge correction.
    solvent_component : openfe.SolventComponent | None
      The SolventComponent of the simulation.

    Returns
    -------
    int
      The alchemical charge difference between the two states.

    Raises
    ------
    ValueError
      * If an explicit charge correction is attempted and the
        nonbonded method is not PME.
      * If the absolute charge difference is greater than one
        and an explicit charge correction is attempted.
      * If an explicit charge correction is attempted and there is no
        solvent present.
    UserWarning
      * If there is any charge difference and no explicit charge
        correction has been requested.
    """
    difference = mapping.get_alchemical_charge_difference()

    if abs(difference) == 0:
        return difference

    if not explicit_charge_correction:
        wmsg = (
            f"A charge difference of {difference} is observed "
            "between the end states. No charge correction has "
            "been requested, please account for this in your "
            "final results."
        )
        logger.warning(wmsg)
        warnings.warn(wmsg)
        return difference

    if solvent_component is None:
        errmsg = "Cannot use explicit charge correction without solvent"
        raise ValueError(errmsg)

    # We implicitly check earlier that we have to have pme for a solvated
    # system, so we only need to check the nonbonded method here
    if nonbonded_method.lower() != "pme":
        errmsg = (
            "Explicit charge correction when not using PME is not currently supported."
        )
        raise ValueError(errmsg)

    if abs(difference) > 1:
        errmsg = (
            f"A charge difference of {difference} is observed "
            "between the end states and an explicit charge  "
            "correction has been requested. Unfortunately "
            "only absolute differences of 1 are supported."
        )
        raise ValueError(errmsg)

    ion = {-1: solvent_component.positive_ion, 1: solvent_component.negative_ion}[
        difference
    ]

    wmsg = (
        f"A charge difference of {difference} is observed "
        "between the end states. This will be addressed by "
        f"transforming a water into a {ion} ion"
    )
    logger.info(wmsg)

    return difference
