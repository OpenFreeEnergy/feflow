"""Test related to the protein mutation protocol and utility functions"""

from importlib.resources import files


class TestProtocolMutation:
    def test_protocol(self):
        return NotImplementedError


def test_mutation_utility():
    """
    Test utility function that creates two chemical systems for the protein mutation protocol.

    It takes an Alanine dipeptide and mutate it to a Proline, generating the corresponding chemical
    systems and mapping objects.

    Returns
    -------

    """
    from gufe import ProteinComponent

    pdb_path = files("feflow.tests.data").joinpath("ALA_capped.pdb")
    # Want to mutate ALA in ACE-ALA-NME seq.
    orig_resname = "ALA"
    target_resname = "PRO"
    resid = 2
    chain_id = 1
    mutation_spec = f"{orig_resname}{resid}{target_resname}"

    # Generate components for mutation
    comp_a, comp_b = some_utility_function(pdb_path=pdb_path, mutation_spec=mutation_spec, chain_id=chain_id)
    omm_top_a = comp_a.to_openmm_topology()
    omm_top_a = comp_b.to_openmm_topology()

    assert omm_top_a == state_b, "Define message here!"

    # Create mapping object
    mapping_obj = some_mapping_function(state_a, state_b)


