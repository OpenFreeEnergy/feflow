"""
Utility functions that can help validating chemical systems and its components such that they
make sense for protocols to use.
"""

# TODO: Migrate utility functions from openfe to this module
from openfe.protocols.openmm_utils.system_validation import (
    get_alchemical_components as _ofe_get_alchemical_components,
)
from openfe.protocols.openmm_utils.system_validation import (
    validate_solvent as _ofe_validate_solvent,
)
from openfe.protocols.openmm_utils.system_validation import (
    validate_protein as _ofe_validate_protein,
)
from openfe.protocols.openmm_rfe.equil_rfe_methods import (
    _validate_alchemical_components,
)

get_alchemical_components = _ofe_get_alchemical_components
validate_solvent = _ofe_validate_solvent
validate_protein = _ofe_validate_protein
validate_alchemical_components = _validate_alchemical_components


# TODO: Implement function to validate mappings -- comps are the same gufe key compared to state
def validate_mappings(state_a, state_b, mapping):
    """
    Validate that the components in the states and the mapping are the correct ones.
    """
    raise NotImplementedError("Function not implemented.")
