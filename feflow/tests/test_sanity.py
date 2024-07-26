"""
Module for testing production and real-life like systems for sanity checks.

This includes systems such as host-guest systems convergence for charge
transformations for both A->B and B->A, among others.
"""

import json
import pytest


@pytest.mark.slow
def test_roundtrip_charge_transformation(tmp_path):
    """
    Run NonEquilibrium Cycling protocol on host-guest charge-changing transformation
    (CB7:A1->A2 and CB7:A2->A1), making sure free energies are equal and opposite, for
    the round trip.
    """
    from importlib.resources import files
    import numpy as np
    from gufe import (
        SmallMoleculeComponent,
        SolventComponent,
        ChemicalSystem,
    )
    from gufe.mapping import LigandAtomMapping
    from gufe.protocols import execute_DAG
    from feflow.protocols import NonEquilibriumCyclingProtocol

    data_basepath = files("feflow.data.host-guest")
    with open(data_basepath.joinpath("cb7_comp.json")) as receptor_file:
        receptor_comp = SmallMoleculeComponent.from_dict(json.load(receptor_file))
    with open(data_basepath.joinpath("a1_comp.json")) as guest1_file:
        guest_1_comp = SmallMoleculeComponent.from_dict(json.load(guest1_file))
    with open(data_basepath.joinpath("a2_comp.json")) as guest2_file:
        guest_2_comp = SmallMoleculeComponent.from_dict(json.load(guest2_file))
    solvent_comp = SolventComponent(positive_ion="Na", negative_ion="Cl")
    state_a_complex = ChemicalSystem(
        {"receptor": receptor_comp, "ligand": guest_1_comp, "solvent": solvent_comp}
    )
    state_b_complex = ChemicalSystem(
        {"receptor": receptor_comp, "ligand": guest_2_comp, "solvent": solvent_comp}
    )
    state_a_solvent = ChemicalSystem({"ligand": guest_1_comp, "solvent": solvent_comp})
    state_b_solvent = ChemicalSystem({"ligand": guest_2_comp, "solvent": solvent_comp})

    mapping_dict = {
        11: 11,
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
        10: 1,
    }

    mapping_obj = LigandAtomMapping(
        componentA=guest_1_comp,
        componentB=guest_2_comp,
        componentA_to_componentB=mapping_dict,
    )
    mapping_reverse_obj = LigandAtomMapping(
        componentA=guest_2_comp,
        componentB=guest_1_comp,
        componentA_to_componentB={
            value: key for key, value in mapping_dict.items()
        },  # reverse
    )

    # Protocol settings
    settings = NonEquilibriumCyclingProtocol.default_settings()
    # Enable charge correction
    settings.alchemical_settings.explicit_charge_correction = True
    protocol = NonEquilibriumCyclingProtocol(settings)

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
    forward_fe_estimate = (
        complex_results.get_estimate() - solvent_results.get_estimate()
    )
    reverse_fe_estimate = (
        complex_reverse_results.get_estimate() - solvent_reverse_results.get_estimate()
    )
    # reverse estimate should have opposite sign so they should add up to close to zero
    forward_reverse_sum = abs(forward_fe_estimate + reverse_fe_estimate)
    forward_reverse_sum_error = np.sqrt(
        complex_results.get_uncertainty() ** 2
        + solvent_results.get_uncertainty() ** 2
        + complex_reverse_results.get_uncertainty() ** 2
        + solvent_reverse_results.get_uncertainty() ** 2
    )
    print(
        f"DDG: {forward_reverse_sum}, 6*dDDG: {6 * forward_reverse_sum_error}"
    )  # debug control print
    assert (
        forward_reverse_sum < 6 * forward_reverse_sum_error
    ), f"DDG ({forward_reverse_sum}) is greater than 6 * dDDG ({6 * forward_reverse_sum_error})"
