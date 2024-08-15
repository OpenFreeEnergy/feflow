"""
Module with mapping objects that are useful for the protein mutation protocol.
"""
import re
from gufe import ProteinComponent
from gufe.mapping import AtomMapping

# TODO: Find a better place for this pattern string, maybe feflow.utils?
regex_mut_str = r"^[a-zA-Z]{3}-?\d+-?[a-zA-Z]{3}$"


class ProteinMutationMapping(AtomMapping):
    """
    Container for an atom mapping for single residue mutations in proteins.

    This is a specialized version of :class:`.AtomMapping` for
    :class:`.ProteinComponent` which stores the mapping as a dict of
    integers.
    """
    componentA: ProteinComponent
    componentB: ProteinComponent
    _mutation_spec: re.Match[regex_mut_str]
    _compA_to_compB: dict[int, int]

    def __init__(
            self,
            componentA: ProteinComponent,
            componentB: ProteinComponent,
            componentA_to_componentB: dict[int, int],
            mutation_spec: re.Match[regex_mut_str],
    ):
        """
        Parameters
        ----------
        componentA, componentB : ProteinComponent
          the protein molecules on either end of the mapping
        componentA_to_componentB : dict[int, int]
          correspondence of indices of atoms between wild type and mutation; the
          keys are indices in componentA and the values are indices in
          componentB.
          These are checked that they are within the possible indices of the
          respective components.
        mutation_spec: str
          Mutation specification string. E.g. "LYS42ALA".
        """
        super().__init__(componentA, componentB)

        # validate compA_to_compB
        nA = self.componentA.to_openmm_topology().getNumAtoms()
        nB = self.componentB.to_openmm_topology().getNumAtoms()
        for i, j in componentA_to_componentB.items():
            if not (0 <= i < nA):
                raise ValueError(f"Got invalid index for ComponentA ({i}); "
                                 f"must be 0 <= n < {nA}")
            if not (0 <= j < nB):
                raise ValueError(f"Got invalid index for ComponentB ({i}); "
                                 f"must be 0 <= n < {nB}")

        self._compA_to_compB = componentA_to_componentB
        self._mutation_spec = mutation_spec

    @property
    def componentA_to_componentB(self) -> dict[int, int]:
        return dict(self._compA_to_compB)

    @property
    def componentB_to_componentA(self) -> dict[int, int]:
        return {v: k for k, v in self._compA_to_compB.items()}

    @property
    def componentA_unique(self):
        return (i for i in range(self.componentA.to_openmm_topology().getNumAtoms())
                if i not in self._compA_to_compB)

    @property
    def componentB_unique(self):
        return (i for i in range(self.componentB.to_openmm_topology().getNumAtoms())
                if i not in self._compA_to_compB.values())

    @property
    def mutation_spec(self):
        """String with the specification of the mutation."""
        return self._mutation_spec

    @classmethod
    def _defaults(cls):
        return {}

    def _to_dict(self) -> dict:
        """Seralize to dictionary"""
        return {
            "componentA": self.componentA,
            "componentB": self.componentB,
            "componentA_to_componentB": self._compA_to_compB,
            "mutation_spec": self.mutation_spec,
        }

    @classmethod
    def _from_dict(cls, dct: dict):
        """Deserialize from dictionary"""
        mapping = dct["componentA_to_componentB"]
        fixed = {int(k): int(v) for k, v in mapping.items()}

        return cls(
            componentA=dct["componentA"],
            componentB=dct["componentB"],
            componentA_to_componentB=fixed,
            mutation_spec=dct["mutation_spec"],
        )