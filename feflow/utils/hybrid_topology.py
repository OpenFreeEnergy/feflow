from openfe.protocols.openmm_rfe._rfe_utils.relative import HybridTopologyFactory


# TODO: This is an utility class. To be deleted when we migrate/extend the base HybridTopologyFactory
class HybridTopologyFactoryModded(HybridTopologyFactory):
    """
    Utility class that extends the base HybridTopologyFactory class with properties for
    getting the indices from initial and final states.
    """

    def __init__(self,
                 old_system, old_positions, old_topology,
                 new_system, new_positions, new_topology,
                 old_to_new_atom_map, old_to_new_core_atom_map,
                 use_dispersion_correction=False,
                 softcore_alpha=0.5,
                 softcore_LJ_v2=True,
                 softcore_LJ_v2_alpha=0.85,
                 interpolate_old_and_new_14s=False,
                 flatten_torsions=False,
                 **kwargs):
        """
        Initialize the Hybrid topology factory.

        Parameters
        ----------
        old_system : openmm.System
            OpenMM system defining the "old" (i.e. starting) state.
        old_positions : [n,3] np.ndarray of float
            The positions of the "old system".
        old_topology : openmm.Topology
            OpenMM topology defining the "old" state.
        new_system: opemm.System
            OpenMM system defining the "new" (i.e. end) state.
        new_positions : [m,3] np.ndarray of float
            The positions of the "new system"
        new_topology : openmm.Topology
            OpenMM topology defining the "new" state.
        old_to_new_atom_map : dict of int : int
            Dictionary of corresponding atoms between the old and new systems.
            Unique atoms are not included in this atom map.
        old_to_new_core_atom_map : dict of int : int
            Dictionary of corresponding atoms between the alchemical "core
            atoms" (i.e. residues which are changing) between the old and
            new systems.
        use_dispersion_correction : bool, default False
            Whether to use the long range correction in the custom sterics
            force. This can be very expensive for NCMC.
        softcore_alpha: float, default None
            "alpha" parameter of softcore sterics, default 0.5.
        softcore_LJ_v2 : bool, default True
            Implement the softcore LJ as defined by Gapsys et al. JCTC 2012.
        softcore_LJ_v2_alpha : float, default 0.85
            Softcore alpha parameter for LJ v2
        interpolate_old_and_new_14s : bool, default False
            Whether to turn off interactions for new exceptions (not just
            1,4s) at lambda = 0 and old exceptions at lambda = 1; if False,
            they are present in the nonbonded force.
        flatten_torsions : bool, default False
            If True, torsion terms involving `unique_new_atoms` will be
            scaled such that at lambda=0,1, the torsion term is turned off/on
            respectively. The opposite is true for `unique_old_atoms`.
        """
        super().__init__(old_system, old_positions, old_topology,
                         new_system, new_positions, new_topology,
                         old_to_new_atom_map, old_to_new_core_atom_map,
                         use_dispersion_correction=False,
                         softcore_alpha=0.5,
                         softcore_LJ_v2=True,
                         softcore_LJ_v2_alpha=0.85,
                         interpolate_old_and_new_14s=False,
                         flatten_torsions=False,
                         **kwargs)

    # TODO: We need to refactor for the init to use these properties and have an attribute with the indices
    @property
    def initial_atom_indices(self):
        """
        Indices of atoms in hybrid topology for atoms in the old topology.

        Returns
        -------
        hybrid_indices: list[int]
            Indices of old atoms in hybrid topology

        """
        n_atoms_old = self._old_system.getNumParticles()
        hybrid_indices = [self._old_to_hybrid_map[idx] for idx in range(n_atoms_old)]
        return hybrid_indices

    @property
    def final_atom_indices(self):
        """
        Indices of atoms in hybrid topology for atoms in the new topology.

        Returns
        -------
        hybrid_indices: list[int]
            Indices of new atoms in hybrid topology

        """
        n_atoms_new = self._new_system.getNumParticles()
        hybrid_indices = [self._new_to_hybrid_map[idx] for idx in range(n_atoms_new)]
        return hybrid_indices
