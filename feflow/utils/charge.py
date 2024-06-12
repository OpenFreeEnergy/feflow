"""
Module involving all the necessary auxiliary utilities and functions for manipulating charges.
Such as assigning both formal and partial charges, or transforming solvent into ions
or vice versa for charge-changing alchemical transformations.
"""

from openfe.protocols.openmm_utils.charge_generation import assign_offmol_partial_charges
from openfe.protocols.openmm_rfe.equil_rfe_methods import _get_alchemical_charge_difference

# TODO: Importing from OpenFE for now, should we migrate them here?
assign_offmol_partial_charges = assign_offmol_partial_charges
get_alchemical_charge_difference = _get_alchemical_charge_difference

