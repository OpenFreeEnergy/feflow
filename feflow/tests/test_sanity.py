"""
Module for testing production and real-life like systems for sanity checks.

This includes systems such as host-guest systems convergence for charge
transformations for both A->B and B->A, among others.
"""

## TODO: HEAVY WIP!

from gufe import ProteinComponent, SmallMoleculeComponent, SolventComponent, ChemicalSystem
from rdkit import Chem
from kartograf import KartografAtomMapper


# In[2]:


receptor = Chem.MolFromMol2File("cb7.sybyl.mol2", removeHs=False)
receptor_comp = SmallMoleculeComponent.from_rdkit(receptor)
guest_1 = Chem.MolFromMol2File("a1.sybyl.mol2", removeHs=False)
#guest_2 = Chem.MolFromMol2File("/home/user/workdir/repos/perses/perses/data/host-guest/a2.sybyl.mol2", removeHs=False)
guest_1_comp = SmallMoleculeComponent.from_rdkit(guest_1)
#guest_2_comp = SmallMoleculeComponent.from_rdkit(guest_2)
guest_2_comp = SmallMoleculeComponent.from_sdf_file("a2.sybyl.sdf")
state_a = ChemicalSystem({"receptor": receptor_comp, "ligand": guest_1_comp})
solvent_comp = SolventComponent(positive_ion="Na", negative_ion="Cl")
state_a_complex = ChemicalSystem({"receptor": receptor_comp, "ligand": guest_1_comp, "solvent": solvent_comp})
state_b_complex = ChemicalSystem({"receptor": receptor_comp, "ligand": guest_2_comp, "solvent": solvent_comp})
state_b_solvent = ChemicalSystem({"ligand": guest_2_comp, "solvent": solvent_comp})
state_a_solvent = ChemicalSystem({"ligand": guest_1_comp, "solvent": solvent_comp})


# In[3]:


mapper = KartografAtomMapper(atom_map_hydrogens=True)
mapping = next(mapper.suggest_mappings(guest_1_comp, guest_2_comp))
mapping


# In[4]:


mapping.componentA_to_componentB


# In[5]:


from gufe.mapping import LigandAtomMapping
mapping_obj = LigandAtomMapping(componentA=guest_1_comp,
                                componentB=guest_2_comp,
                                componentA_to_componentB=mapping.componentA_to_componentB)


# In[7]:


# Protocol settings
from feflow.protocols import NonEquilibriumCyclingProtocol
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol


# In[8]:


default_settings = NonEquilibriumCyclingProtocol.default_settings()
# default_settings = RelativeHybridTopologyProtocol.default_settings()

protocol = NonEquilibriumCyclingProtocol(default_settings)
# protocol = RelativeHybridTopologyProtocol(settings=default_settings)
default_settings


# In[9]:


solvent_dag = protocol.create(stateA=state_a_solvent,
                              stateB=state_b_solvent,
                              name="Host guest solvent leg",
                              mapping=mapping_obj)


# In[10]:


from gufe.protocols import execute_DAG
from pathlib import Path
working_dir = Path(".")
solvent_dag_result = execute_DAG(solvent_dag, shared_basedir=working_dir, scratch_basedir=working_dir)