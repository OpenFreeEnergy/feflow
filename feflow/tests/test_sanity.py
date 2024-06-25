"""
Module for testing production and real-life like systems for sanity checks.

This includes systems such as host-guest systems convergence for charge
transformations for both A->B and B->A, among others.
"""

from openff.units import unit

# TODO: Use openfe analysis tools and functions instead
def ki_to_dg(
        ki: unit.Quantity, uncertainty: unit.Quantity,
        temperature: unit.Quantity = 298.15 * unit.kelvin
) -> tuple[unit.Quantity, unit.Quantity]:
    """
    Convenience method to convert a Ki w/ a given uncertainty to an
    experimental estimate of the binding free energy.

    Parameters
    ----------
    ki : unit.Quantity
        Experimental Ki value (e.g. 5 * unit.nanomolar)
    uncertainty : unit.Quantity
        Experimental error. Note: returns 0 if =< 0 * unit.nanomolar.
    temperature : unit.Quantity
        Experimental temperature. Default: 298.15 * unit.kelvin.

    Returns
    -------
    DG : unit.Quantity
        Gibbs binding free energy.
    dDG : unit.Quantity
        Error in binding free energy.
    """
    import math
    if ki > 1e-15 * unit.nanomolar:
        DG = (unit.molar_gas_constant * temperature.to(unit.kelvin)
              * math.log(ki / unit.molar)).to(unit.kilocalorie_per_mole)
    else:
        raise ValueError("negative Ki values are not supported")
    if uncertainty > 0 * unit.molar:
        dDG = (unit.molar_gas_constant * temperature.to(unit.kelvin)
               * uncertainty / ki).to(unit.kilocalorie_per_mole)
    else:
        dDG = 0 * unit.kilocalorie_per_mole

    return DG, dDG


def test_roundtrip_charge_transformation(tmp_path):
    """
    Run NonEquilibrium Cycling protocol on host-guest charge-changing transformation
    (CB7:A1->A2 and CB7:A2->A1), making sure free energies are equal and opposite, for
    the round trip.
    """
    import numpy as np
    from rdkit import Chem
    from gufe import (
        SmallMoleculeComponent,
        SolventComponent,
        ChemicalSystem,
    )
    from gufe.mapping import LigandAtomMapping
    from gufe.protocols import execute_DAG
    from feflow.protocols import NonEquilibriumCyclingProtocol

    base_data_dir = "/home/user/workdir/repos/perses/perses/data/host-guest/"
    receptor = Chem.MolFromMol2File(f"{base_data_dir}/cb7.sybyl.mol2", removeHs=False)
    receptor_comp = SmallMoleculeComponent.from_rdkit(receptor)
    guest_1 = Chem.MolFromMol2File(f"{base_data_dir}/a1.sybyl.mol2", removeHs=False)
    guest_1_comp = SmallMoleculeComponent.from_rdkit(guest_1)
    guest_2_comp = SmallMoleculeComponent.from_sdf_file(f"{base_data_dir}/a2.sybyl.sdf")
    solvent_comp = SolventComponent(positive_ion="Na", negative_ion="Cl")
    state_a_complex = ChemicalSystem(
        {"receptor": receptor_comp, "ligand": guest_1_comp, "solvent": solvent_comp}
    )
    state_b_complex = ChemicalSystem(
        {"receptor": receptor_comp, "ligand": guest_2_comp, "solvent": solvent_comp}
    )
    state_a_solvent = ChemicalSystem({"ligand": guest_1_comp, "solvent": solvent_comp})
    state_b_solvent = ChemicalSystem({"ligand": guest_2_comp, "solvent": solvent_comp})

    mapping_dict = {11: 11,
                    12: 12,
                    13: 13,
                    14: 14,
                    15: 15,
                    16: 16,
                    17: 17,
                    18: 18,
                    19: 19,
                    20: 20,
                    21: 21,
                    22: 22,
                    23: 23,
                    24: 24,
                    25: 25,
                    26: 26,
                    0: 0,
                    1: 2,
                    2: 3,
                    3: 4,
                    4: 5,
                    5: 6,
                    6: 7,
                    7: 8,
                    8: 9,
                    9: 10,
                    10: 1}

    mapping_obj = LigandAtomMapping(
        componentA=guest_1_comp,
        componentB=guest_2_comp,
        componentA_to_componentB=mapping_dict,
    )
    mapping_reverse_obj = LigandAtomMapping(
        componentA=guest_2_comp,
        componentB=guest_1_comp,
        componentA_to_componentB={value: key for key,value in mapping_dict.items()},  # reverse
    )

    # Protocol settings
    default_settings = NonEquilibriumCyclingProtocol.default_settings()
    ## Debugging settings
    # default_settings.integrator_settings.equilibrium_steps = 1000
    # default_settings.integrator_settings.nonequilibrium_steps = 1000
    # default_settings.work_save_frequency = 500
    # default_settings.num_cycles = 2
    ###
    protocol = NonEquilibriumCyclingProtocol(default_settings)

    # Create DAGs for all the options
    solvent_dag = protocol.create(
        stateA=state_a_solvent,
        stateB=state_b_solvent,
        name="Host-guest solvent forward",
        mapping=mapping_obj,
    )
    solvent_reverse_dag = protocol.create(
        stateA=state_b_solvent,
        stateB=state_a_solvent,
        name="Host-guest solvent reverse",
        mapping=mapping_reverse_obj,
    )
    complex_dag = protocol.create(
        stateA=state_a_complex,
        stateB=state_b_complex,
        name="Host-guest complex forward",
        mapping=mapping_obj,
    )
    complex_reverse_dag = protocol.create(
        stateA=state_b_complex,
        stateB=state_a_complex,
        name="Host-guest complex reverse",
        mapping=mapping_reverse_obj,
    )
    # Execute DAGs
    solvent_dag_result = execute_DAG(
        solvent_dag, shared_basedir=tmp_path, scratch_basedir=tmp_path
    )
    solvent_reverse_dag_result = execute_DAG(
        solvent_reverse_dag, shared_basedir=tmp_path, scratch_basedir=tmp_path
    )
    complex_dag_result = execute_DAG(
        complex_dag, shared_basedir=tmp_path, scratch_basedir=tmp_path
    )
    complex_reverse_dag_result = execute_DAG(
        complex_reverse_dag, shared_basedir=tmp_path, scratch_basedir=tmp_path
    )

    # Compute estimates and compare forward with reverse
    solvent_results = protocol.gather([solvent_dag_result])
    solvent_reverse_results = protocol.gather([solvent_reverse_dag_result])
    complex_results = protocol.gather([complex_dag_result])
    complex_reverse_results = protocol.gather([complex_reverse_dag_result])
    forward_fe_estimate = complex_results.get_estimate() - solvent_results.get_estimate()
    reverse_fe_estimate = complex_reverse_results.get_estimate() - solvent_reverse_results.get_estimate()
    # reverse estimate should have opposite sign so they should add up to close to zero
    forward_reverse_sum = abs(forward_fe_estimate + reverse_fe_estimate)
    forward_reverse_sum_error = np.sqrt(
        complex_results.get_uncertainty()** 2 + solvent_results.get_uncertainty()** 2 +
        complex_reverse_results.get_uncertainty()** 2 + solvent_reverse_results.get_uncertainty()** 2)
    print(f"DDG: {forward_reverse_sum}, 6*dDDG: {6 * forward_reverse_sum_error}")  # debug control print
    assert forward_reverse_sum < 6 * forward_reverse_sum_error, (f"DDG ({forward_reverse_sum}) is greater than 6 * dDDG ({6 * forward_reverse_sum_error})")
