# fixtures for chemicalcomponents and chemicalsystems to test protocols with
import gufe
import pytest
from importlib.resources import files, as_file
from rdkit import Chem
from gufe.mapping import LigandAtomMapping


def pytest_addoption(parser):
    """
    Enables cli argument to run specifically marked tests.
    """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    """Add custom markers to config"""
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Skips tests marked with slow unless flag is specified."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def gufe_data_dir():
    path = files("gufe.tests.data")
    return path


@pytest.fixture(scope="session")
def benzene_modifications(gufe_data_dir):
    source = gufe_data_dir.joinpath("benzene_modifications.sdf")
    with as_file(source) as f:
        supp = Chem.SDMolSupplier(str(f), removeHs=False)
        mols = list(supp)

    return {m.GetProp("_Name"): m for m in mols}


# Components fixtures


@pytest.fixture(scope="session")
def solvent_comp():
    yield gufe.SolventComponent(positive_ion="Na", negative_ion="Cl")


@pytest.fixture(scope="session")
def benzene(benzene_modifications):
    return gufe.SmallMoleculeComponent(benzene_modifications["benzene"])


@pytest.fixture(scope="session")
def toluene(benzene_modifications):
    return gufe.SmallMoleculeComponent(benzene_modifications["toluene"])


# Systems fixtures


@pytest.fixture
def benzene_vacuum_system(benzene):
    return gufe.ChemicalSystem({"ligand": benzene})


@pytest.fixture
def benzene_solvent_system(benzene, solvent_comp):
    return gufe.ChemicalSystem({"ligand": benzene, "solvent": solvent_comp})


@pytest.fixture
def toluene_vacuum_system(toluene):
    return gufe.ChemicalSystem({"ligand": toluene})


@pytest.fixture
def toluene_solvent_system(toluene, solvent_comp):
    return gufe.ChemicalSystem({"ligand": toluene, "solvent": solvent_comp})


# Settings fixtures


@pytest.fixture
def short_settings():
    from openff.units import unit
    from feflow.protocols import NonEquilibriumCyclingProtocol

    settings = NonEquilibriumCyclingProtocol.default_settings()

    settings.thermo_settings.temperature = 300 * unit.kelvin
    settings.integrator_settings.equilibrium_steps = 250
    settings.integrator_settings.nonequilibrium_steps = 250
    settings.work_save_frequency = 50
    settings.traj_save_frequency = 250

    return settings


@pytest.fixture
def short_settings_gpu(short_settings):
    settings = short_settings.copy(deep=True)
    settings.engine_settings.compute_platform = "CUDA"

    return settings


@pytest.fixture
def short_settings_multiple_cycles():
    from openff.units import unit
    from feflow.protocols import NonEquilibriumCyclingProtocol

    settings = NonEquilibriumCyclingProtocol.default_settings()

    settings.thermo_settings.temperature = 300 * unit.kelvin
    settings.integrator_settings.equilibrium_steps = 1000
    settings.integrator_settings.nonequilibrium_steps = 1000
    settings.work_save_frequency = 50
    settings.traj_save_frequency = 250
    settings.num_cycles = 5
    settings.engine_settings.compute_platform = "CPU"

    return settings


@pytest.fixture
def short_settings_multiple_cycles_gpu(short_settings_multiple_cycles):
    settings = short_settings_multiple_cycles.copy(deep=True)
    settings.engine_settings.compute_platform = "CUDA"

    return settings


@pytest.fixture
def production_settings(short_settings):
    settings = short_settings.copy(deep=True)

    settings.eq_steps = 12500
    settings.neq_steps = 12500
    settings.work_save_frequency = 500
    settings.traj_save_frequency = 2000
    settings.num_cycles = 100

    return settings


# Mappings fixtures


@pytest.fixture(scope="session")
def mapping_benzene_toluene(benzene, toluene):
    """Mapping from benzene to toluene"""
    mapping_benzene_to_toluene = {
        0: 4,
        1: 5,
        2: 6,
        3: 7,
        4: 8,
        5: 9,
        6: 10,
        7: 11,
        8: 12,
        9: 13,
        11: 14,
    }
    mapping_obj = LigandAtomMapping(
        componentA=benzene,
        componentB=toluene,
        componentA_to_componentB=mapping_benzene_to_toluene,
    )
    return mapping_obj


@pytest.fixture
def mapping_toluene_toluene(toluene):
    """Mapping from toluene to toluene"""
    mapping_toluene_to_toluene = {
        i: i for i in range(len(toluene.to_rdkit().GetAtoms()))
    }
    mapping_obj = LigandAtomMapping(
        componentA=toluene,
        componentB=toluene,
        componentA_to_componentB=mapping_toluene_to_toluene,
    )
    return mapping_obj


@pytest.fixture
def broken_mapping(benzene, toluene):
    """Broken mapping"""
    # Mapping that doesn't make sense for benzene and toluene
    broken_mapping = {
        40: 20,
        5: 1,
        6: 2,
        7: 3,
        38: 4,
        9: 5,
        10: 6,
        191: 7,
        12: 8,
        13: 99,
        14: 11,
    }
    broken_mapping_obj = LigandAtomMapping(
        componentA=benzene,
        componentB=toluene,
        componentA_to_componentB=broken_mapping,
    )
    return broken_mapping_obj
