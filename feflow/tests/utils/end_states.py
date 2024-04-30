import copy
from openmm import unit as omm_unit


def check_system(system):
    """
    Check OpenMM System object for pathologies, like duplicate atoms in torsions.

    Parameters
    ----------
    system : openmm.System

    """
    # from openmm import XmlSerializer
    forces = {
        system.getForce(index).__class__.__name__: system.getForce(index)
        for index in range(system.getNumForces())
    }
    force = forces["PeriodicTorsionForce"]
    for index in range(force.getNumTorsions()):
        [i, j, k, l, _, _, _] = force.getTorsionParameters(index)
        if len({i, j, k, l}) < 4:
            msg = f"Torsion index {index} of self._topology_proposal.new_system has duplicate atoms: {i} {j} {k} {l}\n"
            msg += "Serialized system to system.xml for inspection.\n"
            raise Exception(msg)
    # IP: I don't think we need to serialize
    # serialized_system = XmlSerializer.serialize(system)
    # outfile = open('system.xml', 'w')
    # outfile.write(serialized_system)
    # outfile.close()


def generate_endpoint_thermodynamic_states(
    system,
    topology_proposal,
    repartitioned_endstate=None,
    temperature=300.0 * omm_unit.kelvin,
):
    """
    Generate endpoint thermodynamic states for the system

    Parameters
    ----------
    system : openmm.System
        System object corresponding to thermodynamic state
    topology_proposal : perses.rjmc.topology_proposal.TopologyProposal
        TopologyProposal representing transformation
    repartitioned_endstate : int, default None
        If the htf was generated using RepartitionedHybridTopologyFactory, use this argument to
        specify the endstate at which it was generated. Otherwise, leave as None.
    temperature : openmm.unit.Quantity, default 300 K
        Temperature to set when generating the thermodynamic states

    Returns
    -------
    nonalchemical_zero_thermodynamic_state : ThermodynamicState
        Nonalchemical thermodynamic state for lambda zero endpoint
    nonalchemical_one_thermodynamic_state : ThermodynamicState
        Nonalchemical thermodynamic state for lambda one endpoint
    lambda_zero_thermodynamic_state : ThermodynamicState
        Alchemical (hybrid) thermodynamic state for lambda zero
    lambda_one_thermodynamic_State : ThermodynamicState
        Alchemical (hybrid) thermodynamic state for lambda one
    """
    # Create the thermodynamic state
    from feflow.utils.lambda_protocol import RelativeAlchemicalState
    from openmmtools import states

    check_system(system)

    # Create thermodynamic states for the nonalchemical endpoints
    nonalchemical_zero_thermodynamic_state = states.ThermodynamicState(
        topology_proposal.old_system, temperature=temperature
    )
    nonalchemical_one_thermodynamic_state = states.ThermodynamicState(
        topology_proposal.new_system, temperature=temperature
    )

    # Create the base thermodynamic state with the hybrid system
    thermodynamic_state = states.ThermodynamicState(system, temperature=temperature)

    if repartitioned_endstate == 0:
        lambda_zero_thermodynamic_state = thermodynamic_state
        lambda_one_thermodynamic_state = None
    elif repartitioned_endstate == 1:
        lambda_zero_thermodynamic_state = None
        lambda_one_thermodynamic_state = thermodynamic_state
    else:
        # Create relative alchemical states
        lambda_zero_alchemical_state = RelativeAlchemicalState.from_system(system)
        lambda_one_alchemical_state = copy.deepcopy(lambda_zero_alchemical_state)

        # Ensure their states are set appropriately
        lambda_zero_alchemical_state.set_alchemical_parameters(0.0)
        lambda_one_alchemical_state.set_alchemical_parameters(1.0)

        # Now create the compound states with different alchemical states
        lambda_zero_thermodynamic_state = states.CompoundThermodynamicState(
            thermodynamic_state, composable_states=[lambda_zero_alchemical_state]
        )
        lambda_one_thermodynamic_state = states.CompoundThermodynamicState(
            thermodynamic_state, composable_states=[lambda_one_alchemical_state]
        )

    return (
        nonalchemical_zero_thermodynamic_state,
        nonalchemical_one_thermodynamic_state,
        lambda_zero_thermodynamic_state,
        lambda_one_thermodynamic_state,
    )
