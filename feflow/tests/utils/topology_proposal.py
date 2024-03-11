"""
Utility module for tests to process perses TopologyProposal objects to extract objects useful
for FEFlow/OpenFE
"""

from perses.rjmc.topology_proposal import TopologyProposal


def extract_htf_data(top_proposal: TopologyProposal):
    """
    Extract OpenMM system and OpenMM topology data objects from a perses TopologyProposal object.
    In order to be passed to the HybridTopologyFactory constructor.

    Parameters
    ----------
    top_proposal: perses.rjmc.topology_proposal.TopologyProposal
        Instance of TopologyProposal class from perses where to extract the data from.

    Returns
    -------
    htf_data: dict
        Dictionary with the data for the HybridTopologyFactory constructor.
        Keys are "old/new_system", "old/new_topology".
    """
    # Extract systems
    old_system = top_proposal.old_system
    new_system = top_proposal.new_system
    # Extract coordinates
    old_topology = top_proposal.old_topology
    new_topology = top_proposal.new_topology
    # Extract atom maps
    old_to_new_atom_map = top_proposal.old_to_new_atom_map
    # TODO: Check that core atoms are understood as the same in Perses. I'm not sure they are.
    old_to_new_core_atom_map = {
        value: key for key, value in top_proposal.core_new_to_old_atom_map.items()
    }

    htf_data = {
        "old_system": old_system,
        "new_system": new_system,
        "old_topology": old_topology,
        "new_topology": new_topology,
        "old_to_new_atom_map": old_to_new_atom_map,
        "old_to_new_core_atom_map": old_to_new_core_atom_map,
    }

    return htf_data
