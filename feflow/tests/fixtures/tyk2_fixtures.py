"""
Fixtures related to tyk2 benchmarking system. Both proteins and ligands.
"""

from importlib.resources import files

import gufe
import pytest


@pytest.fixture(scope="session")
def tyk2_protein():
    filepath = files("feflow.tests.data.protein_ligand").joinpath("tyk2_protein.pdb")
    return gufe.ProteinComponent.from_pdb_file(str(filepath))


@pytest.fixture(scope="session")
def tyk2_ligand_ejm_54():
    filepath = files("feflow.tests.data.protein_ligand").joinpath("tyk2_lig_ejm_54.sdf")
    return gufe.SmallMoleculeComponent.from_sdf_file(str(filepath))


@pytest.fixture(scope="session")
def tyk2_ligand_ejm_46():
    filepath = files("feflow.tests.data.protein_ligand").joinpath("tyk2_lig_ejm_46.sdf")
    return gufe.SmallMoleculeComponent.from_sdf_file(str(filepath))


@pytest.fixture(scope="session")
def tyk2_lig_ejm_31():
    input_sdf = files("feflow.tests.data.protein_ligand").joinpath(
        "tyk2_lig_ejm_31.sdf"
    )
    return gufe.SmallMoleculeComponent.from_sdf_file(str(input_sdf))


@pytest.fixture(scope="session")
def tyk2_lig_ejm_55():
    input_sdf = files("feflow.tests.data.protein_ligand").joinpath(
        "tyk2_lig_ejm_55.sdf"
    )
    return gufe.SmallMoleculeComponent.from_sdf_file(str(input_sdf))


@pytest.fixture(scope="session")
def tyk2_lig_ejm_46_complex(tyk2_protein, tyk2_ligand_ejm_46, solvent_comp):
    return gufe.ChemicalSystem(
        {"protein": tyk2_protein, "ligand": tyk2_ligand_ejm_46, "solvent": solvent_comp}
    )


@pytest.fixture(scope="session")
def tyk2_lig_ejm_54_complex(tyk2_protein, tyk2_ligand_ejm_54, solvent_comp):
    return gufe.ChemicalSystem(
        {"protein": tyk2_protein, "ligand": tyk2_ligand_ejm_54, "solvent": solvent_comp}
    )


@pytest.fixture(scope="session")
def mapping_tyk2_54_to_46(tyk2_ligand_ejm_54, tyk2_ligand_ejm_46):
    """
    Mapping object from ligand ejm_54 to ejm_46 for the Tyk2 dataset.

    It generates the mapping on runtime using the Kartograf mapper.
    """
    from kartograf import KartografAtomMapper

    atom_mapper = KartografAtomMapper()
    mapping_obj = next(
        atom_mapper.suggest_mappings(tyk2_ligand_ejm_54, tyk2_ligand_ejm_46)
    )

    return mapping_obj


@pytest.fixture(scope="session")
def tyk2_lig_ejm_31_to_lig_ejm_55_mapping(tyk2_lig_ejm_31, tyk2_lig_ejm_55):
    from kartograf import KartografAtomMapper

    atom_mapper = KartografAtomMapper()
    mapping = next(atom_mapper.suggest_mappings(tyk2_lig_ejm_31, tyk2_lig_ejm_55))
    return mapping


@pytest.fixture(scope="session")
def tyk2_ejm_31_to_ejm_55_systems_only_ligands(
    tyk2_lig_ejm_31_to_lig_ejm_55_mapping, solvent_comp
):
    """
    This fixture returns a dictionary with both state A and state B for the tyk2
    lig_ejm_31 to lig_ejm_55 transformation, as chemical systems. The systems are solvated.
    """
    state_a = {
        "ligand": tyk2_lig_ejm_31_to_lig_ejm_55_mapping.componentA,
        "solvent": solvent_comp,
    }
    state_b = {
        "ligand": tyk2_lig_ejm_31_to_lig_ejm_55_mapping.componentB,
        "solvent": solvent_comp,
    }
    system_a = gufe.ChemicalSystem(state_a)
    system_b = gufe.ChemicalSystem(state_b)
    return {"state_a": system_a, "state_b": system_b}


@pytest.fixture(scope="session")
def tyk2_ejm_31_to_ejm_55_systems(
    tyk2_protein, tyk2_lig_ejm_31_to_lig_ejm_55_mapping, solvent_comp
):
    """
    This fixture returns a dictionary with both state A and state B for the tyk2
    lig_ejm_31 to lig_ejm_55 transformation, as chemical systems. The systems are solvated.
    """
    state_a = {
        "protein": tyk2_protein,
        "ligand": tyk2_lig_ejm_31_to_lig_ejm_55_mapping.componentA,
        "solvent": solvent_comp,
    }
    state_b = {
        "protein": tyk2_protein,
        "ligand": tyk2_lig_ejm_31_to_lig_ejm_55_mapping.componentB,
        "solvent": solvent_comp,
    }
    system_a = gufe.ChemicalSystem(state_a)
    system_b = gufe.ChemicalSystem(state_b)
    return {"state_a": system_a, "state_b": system_b}
