# Adapted from perses: https://github.com/choderalab/perses/blob/protocol-neqcyc/perses/protocols/nonequilibrium_cycling.py

from typing import Optional, Iterable, List, Dict, Any

import datetime
import logging
import time

from gufe.settings import Settings, OpenMMSystemGeneratorFFSettings, ThermoSettings
from gufe.chemicalsystem import ChemicalSystem
from gufe.mapping import ComponentMapping
from gufe.protocols import (
    Protocol,
    ProtocolUnit,
    ProtocolResult,
    ProtocolDAGResult,
)

from perses.app.relative_setup import RelativeFEPSetup
from perses.app.setup_relative_calculation import get_openmm_platform
from perses.annihilation.relative import HybridTopologyFactory

from openff.units import unit
from openff.units.openmm import to_openmm

# Specific instance of logger for this module
# logger = logging.getLogger(__name__)


class SimulationUnit(ProtocolUnit):
    """
    Monolithic unit for simulation. It runs NEQ switching simulation from chemical systems and stores the
    work computed in numpy-formatted files, to be analyzed by another unit.
    """
    @staticmethod
    def _check_states_compatibility(state_a, state_b):
        """
        Checks that both states have the same solvent parameters and receptor.

        Parameters
        ----------
        state_a : gufe.state.State
            Origin state for the alchemical transformation.
        state_b :
            Destination state for the alchemical transformation.
        """
        # If any of them has a solvent, check the parameters are the same
        if any(["solvent" in state.components for state in (state_a, state_b)]):
            assert state_a.get("solvent") == state_b.get("solvent"), "Solvent parameters differ between solvent components."
        # check protein component is the same in both states if protein component is found
        if any(["protein" in state.components for state in (state_a, state_b)]):
            assert state_a.get("protein") == state_b.get("protein"), "Receptors in states are not compatible."

    @staticmethod
    def _detect_phase(state_a, state_b):
        """
        Detect phase according to the components in the input chemical state.

        Complex state is assumed if both states have ligands and protein components.

        Solvent state is assumed

        Vacuum state is assumed if only either a ligand or a protein is present
        in each of the states.

        Parameters
        ----------
        state_a : gufe.state.State
            Source state for the alchemical transformation.
        state_b : gufe.state.State
            Destination state for the alchemical transformation.

        Returns
        -------
        phase : str
            Phase name. "vacuum", "solvent" or "complex".
        component_keys : list[str]
            List of component keys to extract from states.
        """
        states = (state_a, state_b)
        # where to store the data to be returned

        # Order of phases is important! We have to check complex first and solvent second.
        key_options = {
            "complex": ["ligand", "protein", "solvent"],
            "solvent": ["ligand", "solvent"],
            "vacuum": ["ligand"]
        }
        for phase, keys in key_options.items():
            if all([key in state for state in states for key in keys]):
                detected_phase = phase
                break
        else:
            raise ValueError(
                "Could not detect phase from system states. Make sure the component in both systems match.")

        return detected_phase

    @staticmethod
    def extract_positions(context, hybrid_topology_factory, atom_selection_exp="not water"):
        """
        Extract positions from initial and final systems based from the hybrid topology.

        Parameters
        ----------
        context: openmm.Context
            Current simulation context where from extract positions.
        hybrid_topology_factory: perses.annihilation.relative.HybridTopologyFactory
            Hybrid topology factory where to extract positions and mapping information
        atom_selection_exp: str, optional
            Atom selection expression using mdtraj syntax. Defaults to "not water"

        Returns
        -------

        Notes
        -----
        It achieves this by taking the positions and indices from the initial and final states of
        the transformation, and computing the overlap of these with the indices of the complete
        hybrid topology, filtered by some mdtraj selection expression.

        1. Get positions from context
        2. Get topology from HTF (already mdtraj topology)
        3. Merge that information into mdtraj.Trajectory
        4. Filter positions for initial/final according to selection string
        """
        # TODO: Maybe we want this as a helper/utils function in perses. We also need tests for this.
        import mdtraj as md
        import numpy as np

        # Get positions from current openmm context
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)

        # Get topology from HTF - indices for initial and final topologies in hybrid topology
        initial_indices = np.asarray(hybrid_topology_factory.old_atom_indices)
        final_indices = np.asarray(hybrid_topology_factory.new_atom_indices)
        hybrid_topology = hybrid_topology_factory.hybrid_topology
        selection = atom_selection_exp
        md_trajectory = md.Trajectory(xyz=positions, topology=hybrid_topology)
        selection_indices = md_trajectory.topology.select(selection)

        # Now we have to find the intersection/overlap between selected indices in the hybrid
        # topology and the initial/final positions, respectively
        initial_selected_indices = np.intersect1d(initial_indices, selection_indices)
        final_selected_indices = np.intersect1d(final_indices, selection_indices)
        initial_selected_positions = md_trajectory.xyz[0, initial_selected_indices, :]
        final_selected_positions = md_trajectory.xyz[0, final_selected_indices, :]

        return initial_selected_positions, final_selected_positions

    def _execute(self, ctx, *, state_a, state_b, mapping, settings, **inputs):
        """
        Execute the simulation part of the Nonequilibrium switching protocol using GUFE objects.

        Parameters
        ----------
        ctx: gufe.protocols.protocolunit.Context
            The gufe context for the unit.

        state_a : gufe.ChemicalSystem
            The initial chemical system.

        state_b : gufe.ChemicalSystem
            The objective chemical system.

        mapping : dict[str, gufe.mapping.ComponentMapping]
            A dict featuring mappings between the two chemical systems.

        settings : gufe.settings.model.Settings
            The full settings for the protocol.

        Returns
        -------
        dict : dict[str, str]
            Dictionary with paths to work arrays, both forward and reverse, and trajectory coordinates for systems
            A and B.
        """
        # needed imports
        import numpy as np
        import openmm
        import openmm.unit as openmm_unit
        from openmmtools.integrators import PeriodicNonequilibriumIntegrator
        from perses.utils.openeye import generate_unique_atom_names

        # Setting up logging to file in shared filesystem
        file_logger = logging.getLogger("neq-cycling")
        output_log_path = ctx.shared / "perses-neq-cycling.log"
        file_handler = logging.FileHandler(output_log_path, mode="w")
        file_handler.setLevel(logging.DEBUG)  # TODO: Set to INFO in production
        log_formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(log_formatter)
        file_logger.addHandler(file_handler)

        # Check compatibility between states (same receptor and solvent)
        self._check_states_compatibility(state_a, state_b)

        # Get components from systems if found (None otherwise) -- NOTE: Uses hardcoded keys!
        receptor_a = state_a.components.get("protein")
        # receptor_b = state_b.components.get("protein")  # Should not be needed
        ligand_a = mapping.get("ligand").componentA
        ligand_b = mapping.get("ligand").componentB
        solvent_a = state_a.components.get("solvent")
        # solvent_b = state_b.components.get("solvent")  # Should not be needed


        # Check first state for receptor if not get receptor from second one
        if receptor_a:
            receptor_top = receptor_a.to_openmm_topology()
            receptor_pos = receptor_a.to_openmm_positions()
        else:
            receptor_top, receptor_pos = None, None

        # Get ligands cheminformatics molecules
        ligand_a = ligand_a.to_openff().to_openeye()
        ligand_b = ligand_b.to_openff().to_openeye()
        # Generating unique atom names for ligands -- openmmforcefields needs them
        ligand_a = generate_unique_atom_names(ligand_a)
        ligand_b = generate_unique_atom_names(ligand_b)

        # Get solvent parameters from component
        if solvent_a:
            ion_concentration = solvent_a.ion_concentration.to_openmm()
            positive_ion = solvent_a.positive_ion
            negative_ion = solvent_a.negative_ion
        else:
            ion_concentration, positive_ion, negative_ion = None, None, None

        # Get settings
        thermodynamic_settings = settings.thermo_settings
        phase = self._detect_phase(state_a, state_b)
        traj_save_frequency = settings.traj_save_frequency
        work_save_frequency = settings.work_save_frequency  # Note: this is divisor of traj save freq.
        selection_expression = settings.atom_selection_expression

        # Get the ligand mapping from ComponentMapping object
        # NOTE: perses to date has a different "directionality" sense in terms of the mapping,
        #   see perses.rjmc.topology_proposal.propose docstring for detailed information.
        ligand_mapping = mapping['ligand'].componentB_to_componentA

        # Setup relative FE calculation
        fe_setup = RelativeFEPSetup(
            old_ligand=ligand_a,
            new_ligand=ligand_b,
            receptor=receptor_top,
            receptor_positions=receptor_pos,
            forcefield_files=settings.forcefield_settings.forcefields,
            small_molecule_forcefield=settings.forcefield_settings.small_molecule_forcefield,
            phases=[phase],
            transformation_atom_map=ligand_mapping,  # Handle atom mapping between systems
            ionic_strength=ion_concentration,
            positive_ion=positive_ion,
            negative_ion=negative_ion,
        )

        topology_proposals = fe_setup.topology_proposals
        old_positions = fe_setup.old_positions
        new_positions = fe_setup.new_positions

        # Generate Hybrid Topology Factory - Generic HTF
        htf = HybridTopologyFactory(
            topology_proposal=topology_proposals[phase],
            current_positions=old_positions[phase],
            new_positions=new_positions[phase],
            softcore_LJ_v2=settings.softcore_LJ_v2,
            interpolate_old_and_new_14s=settings.interpolate_old_and_new_14s,
        )

        system = htf.hybrid_system
        positions = htf.hybrid_positions

        # Set up integrator
        temperature = to_openmm(thermodynamic_settings.temperature)
        neq_steps = settings.eq_steps
        eq_steps = settings.neq_steps
        timestep = to_openmm(settings.timestep)
        splitting = settings.neq_splitting
        integrator = PeriodicNonequilibriumIntegrator(alchemical_functions=settings.lambda_functions,
                                                      nsteps_neq=neq_steps,
                                                      nsteps_eq=eq_steps,
                                                      splitting=splitting,
                                                      timestep=timestep,
                                                      temperature=temperature, )

        # Set up context
        platform = get_openmm_platform(settings.platform)
        context = openmm.Context(system, integrator, platform)
        context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
        context.setPositions(positions)

        try:
            # Minimize
            openmm.LocalEnergyMinimizer.minimize(context)

            # Equilibrate
            context.setVelocitiesToTemperature(temperature)

            # Prepare objects to store data -- empty lists so far
            forward_eq_old, forward_eq_new, forward_neq_old, forward_neq_new = list(), list(), list(), list()
            reverse_eq_new, reverse_eq_old, reverse_neq_old, reverse_neq_new = list(), list(), list(), list()

            # Coarse number of steps -- each coarse consists of work_save_frequency steps
            coarse_eq_steps = int(eq_steps/work_save_frequency)  # Note: eq_steps is multiple of work save steps
            coarse_neq_steps = int(neq_steps / work_save_frequency)  # Note: neq_steps is multiple of work save steps

            # TODO: Also get the GPU information (plain try-except with nvidia-smi)

            # Equilibrium (lambda = 0)
            # start timer
            start_time = time.perf_counter()
            for step in range(coarse_eq_steps):
                integrator.step(work_save_frequency)
                file_logger.debug(f"coarse step: {step}: saving work (freq {work_save_frequency})")
                # Save positions
                if step % traj_save_frequency == 0:
                    file_logger.debug(f"coarse step: {step}: saving trajectory (freq {traj_save_frequency})")
                    initial_positions, final_positions = self.extract_positions(context, hybrid_topology_factory=htf,
                                                                                atom_selection_exp=selection_expression)
                    forward_eq_old.append(initial_positions)
                    forward_eq_new.append(final_positions)
            # Make sure trajectories are stored at the end of the eq loop
            file_logger.debug(f"coarse step: {step}: saving trajectory (freq {traj_save_frequency})")
            initial_positions, final_positions = self.extract_positions(context, hybrid_topology_factory=htf,
                                                                        atom_selection_exp=selection_expression)
            forward_eq_old.append(initial_positions)
            forward_eq_new.append(final_positions)

            eq_forward_time = time.perf_counter()
            eq_forward_walltime = datetime.timedelta(seconds=eq_forward_time - start_time)
            file_logger.info(f"replicate_{self.name} Forward (lamba = 0) equilibration time : {eq_forward_walltime}")

            # Run neq
            # Forward (0 -> 1)
            # Initialize works with current value
            forward_works = [integrator.get_protocol_work(dimensionless=True)]
            for fwd_step in range(coarse_neq_steps):
                integrator.step(work_save_frequency)
                forward_works.append(integrator.get_protocol_work(dimensionless=True))
                if fwd_step % traj_save_frequency == 0:
                    initial_positions, final_positions = self.extract_positions(context, hybrid_topology_factory=htf,
                                                                                atom_selection_exp=selection_expression)
                    forward_neq_old.append(initial_positions)
                    forward_neq_new.append(final_positions)
            # Make sure trajectories are stored at the end of the neq loop
            initial_positions, final_positions = self.extract_positions(context, hybrid_topology_factory=htf,
                                                                        atom_selection_exp=selection_expression)
            forward_neq_old.append(initial_positions)
            forward_neq_new.append(final_positions)

            neq_forward_time = time.perf_counter()
            neq_forward_walltime = datetime.timedelta(seconds=neq_forward_time - eq_forward_time)
            file_logger.info(f"replicate_{self.name} Forward nonequilibrium time (lambda 0 -> 1): {neq_forward_walltime}")

            # Equilibrium (lambda = 1)
            for step in range(coarse_eq_steps):
                integrator.step(work_save_frequency)
                if step % traj_save_frequency == 0:
                    initial_positions, final_positions = self.extract_positions(context, hybrid_topology_factory=htf,
                                                                                atom_selection_exp=selection_expression)
                    reverse_eq_new.append(initial_positions)  # TODO: Maybe better naming not old/new but initial/final
                    reverse_eq_old.append(final_positions)
            # Make sure trajectories are stored at the end of the eq loop
            initial_positions, final_positions = self.extract_positions(context, hybrid_topology_factory=htf,
                                                                        atom_selection_exp=selection_expression)
            reverse_eq_old.append(initial_positions)
            reverse_eq_new.append(final_positions)

            eq_reverse_time = time.perf_counter()
            eq_reverse_walltime = datetime.timedelta(seconds=eq_reverse_time - neq_forward_time)
            file_logger.info(f"replicate_{self.name} Reverse (lambda 1) time: {eq_reverse_walltime}")

            # Reverse work (1 -> 0)
            reverse_works = [integrator.get_protocol_work(dimensionless=True)]
            for rev_step in range(coarse_neq_steps):
                integrator.step(work_save_frequency)
                reverse_works.append(integrator.get_protocol_work(dimensionless=True))
                if rev_step % traj_save_frequency == 0:
                    initial_positions, final_positions = self.extract_positions(context, hybrid_topology_factory=htf,
                                                                                atom_selection_exp=selection_expression)
                    reverse_neq_old.append(initial_positions)
                    reverse_neq_new.append(final_positions)
            # Make sure trajectories are stored at the end of the neq loop
            initial_positions, final_positions = self.extract_positions(context, hybrid_topology_factory=htf,
                                                                        atom_selection_exp=selection_expression)
            forward_eq_old.append(initial_positions)
            forward_eq_new.append(final_positions)

            neq_reverse_time = time.perf_counter()
            neq_reverse_walltime = datetime.timedelta(seconds=neq_reverse_time - eq_reverse_time)
            file_logger.info(f"replicate_{self.name} Reverse nonequilibrium time (lambda 1 -> 0): {neq_reverse_walltime}")
            # Elapsed time for the whole cycle
            cycle_walltime = datetime.timedelta(seconds=neq_reverse_time - start_time)
            file_logger.info(f"replicate_{self.name} Nonequilibrium cycle total walltime: {cycle_walltime}")

            # Computing performance in ns/day
            simulation_time = 2*(eq_steps + neq_steps)*timestep
            walltime_in_seconds = cycle_walltime.total_seconds() * openmm_unit.seconds
            estimated_performance = simulation_time.value_in_unit(
                openmm_unit.nanosecond) / walltime_in_seconds.value_in_unit(openmm_unit.days)  # in ns/day
            file_logger.info(f"replicate_{self.name} Estimated performance: {estimated_performance} ns/day")

            # Serialize works
            forward_work_path = ctx.shared / f"forward_{phase}_{self.name}.npy"
            reverse_work_path = ctx.shared / f"reverse_{phase}_{self.name}.npy"
            with open(forward_work_path, 'wb') as out_file:
                np.save(out_file, forward_works)
            with open(reverse_work_path, 'wb') as out_file:
                np.save(out_file, reverse_works)

            # Store timings and performance info in dictionary
            timing_info = {
                "eq_forward_time_in_s": eq_forward_walltime.total_seconds(),
                "neq_forward_time_in_s": neq_forward_walltime.total_seconds(),
                "eq_reverse_time_in_s": eq_reverse_walltime.total_seconds(),
                "neq_reverse_time_in_s": neq_reverse_walltime.total_seconds(),
                "performance_in_ns_day": estimated_performance
            }

            # TODO: Do we need to save the trajectories?
            # Serialize trajectories
            forward_eq_old_path = ctx.shared / f"forward_eq_old_{phase}_{self.name}.npy"
            forward_eq_new_path = ctx.shared / f"forward_eq_new_{phase}_{self.name}.npy"
            forward_neq_old_path = ctx.shared / f"forward_neq_old_{phase}_{self.name}.npy"
            forward_neq_new_path = ctx.shared / f"forward_neq_new_{phase}_{self.name}.npy"
            reverse_eq_new_path = ctx.shared / f"reverse_eq_new_{phase}_{self.name}.npy"
            reverse_eq_old_path = ctx.shared / f"reverse_eq_old_{phase}_{self.name}.npy"
            reverse_neq_old_path = ctx.shared / f"reverse_neq_old_{phase}_{self.name}.npy"
            reverse_neq_new_path = ctx.shared / f"reverse_neq_new_{phase}_{self.name}.npy"

            with open(forward_eq_old_path, 'wb') as out_file:
                np.save(out_file, np.array(forward_eq_old))
            with open(forward_eq_new_path, 'wb') as out_file:
                np.save(out_file, np.array(forward_eq_new))
            with open(reverse_eq_old_path, 'wb') as out_file:
                np.save(out_file, np.array(reverse_eq_old))
            with open(reverse_eq_new_path, 'wb') as out_file:
                np.save(out_file, np.array(reverse_eq_new))
            with open(forward_neq_old_path, 'wb') as out_file:
                np.save(out_file, np.array(forward_neq_old))
            with open(forward_neq_new_path, 'wb') as out_file:
                np.save(out_file, np.array(forward_neq_new))
            with open(reverse_neq_old_path, 'wb') as out_file:
                np.save(out_file, np.array(reverse_neq_old))
            with open(reverse_neq_new_path, 'wb') as out_file:
                np.save(out_file, np.array(reverse_neq_new))

            # Saving trajectory paths in dictionary structure
            trajectory_paths = {
                "forward_eq_initial": forward_eq_old_path,
                "forward_eq_final": forward_eq_new_path,
                "forward_neq_initial": forward_neq_old_path,
                "forward_neq_final": forward_neq_new_path,
                "reverse_eq_initial": reverse_eq_old_path,
                "reverse_eq_final": reverse_eq_new_path,
                "reverse_neq_initial": reverse_neq_old_path,
                "reverse_neq_final": reverse_neq_new_path
            }
        finally:
            # Explicit cleanup for GPU resources
            del context, integrator

        return {
            'forward_work': forward_work_path,
            'reverse_work': reverse_work_path,
            "trajectory_paths": trajectory_paths,
            'log': output_log_path,
            'timing_info': timing_info,
        }


class ResultUnit(ProtocolUnit):
    """
    Returns cumulated work and paths for output files.
    """

    @staticmethod
    def _execute(ctx, *, simulations, **inputs):
        import numpy as np
        # TODO: This can take the settings and process a debug flag, and populate all the paths for trajectories as needed
        # Load the works from shared serialized objects
        # TODO: We need to make sure the array is the CUMULATIVE work and that we just want the last value
        forward_works = []
        reverse_works = []
        for simulation in simulations:
            forward_work = np.load(simulation.outputs['forward_work'])
            reverse_work = np.load(simulation.outputs['reverse_work'])

            forward_works.append(forward_work - forward_work[0])
            reverse_works.append(reverse_work - reverse_work[0])

        # Path dictionary
        paths_dict = {"forward_work": [simulation.outputs["forward_work"] for simulation in simulations],
                      "reverse_work": [simulation.outputs["reverse_work"] for simulation in simulations]}

        return {"forward_work": forward_works,
                "reverse_work": reverse_works,
                "paths": paths_dict,
                }


class NonEquilibriumCyclingProtocolResult(ProtocolResult):
    """
    Gathers results from different runs and computes the free energy estimates using BAR and its errors using
    bootstrapping.
    """

    def get_estimate(self):
        """
        Get a free energy estimate using bootstrap and BAR.

        Parameters
        ----------
        n_bootstraps: int
            Number of bootstrapped samples to use.

        Returns
        -------
        free_energy: float
            Free energy estimate in units of kT.

        """
        import numpy as np
        import numpy.typing as npt
        import pymbar

        forward_work = [i[-1] for i in self.data['forward_work']]
        reverse_work = [i[-1] for i in self.data['reverse_work']]

        forward_work: npt.NDArray[float] = np.array(forward_work)
        reverse_work: npt.NDArray[float] = np.array(reverse_work)
        free_energy, error = pymbar.bar.BAR(forward_work, reverse_work)

        return (free_energy * unit.k * self.data['temperature'] * unit.avogadro_constant).to('kcal/mol')

    def get_uncertainty(self, n_bootstraps=1000):
        """
        Estimate uncertainty using standard deviation of the distribution of bootstrapped
        free energy (dg) samples.

        Parameters
        ----------
        n_bootstraps

        Returns
        -------
        free_energy_uncertainty: float
            Uncertainty on the free energy estimate in units of kT.

        """
        import numpy as np
        import numpy.typing as npt

        forward_work = [i[-1] for i in self.data['forward_work']]
        reverse_work = [i[-1] for i in self.data['reverse_work']]

        forward: npt.NDArray[float] = np.array(forward_work)
        reverse: npt.NDArray[float] = np.array(reverse_work)

        all_dgs = self._do_bootstrap(forward, reverse, n_bootstraps=n_bootstraps)

        # TODO: Check if standard deviation is a good uncertainty estimator
        return (np.std(all_dgs) * unit.k * self.data['temperature'] * unit.avogadro_constant).to('kcal/mol')

    def get_rate_of_convergence(self):
        ...

    # @lru_cache()
    def _do_bootstrap(self, forward, reverse, n_bootstraps=1000):
        """
        Performs bootstrapping from forward and reverse cumulated works.

        Parameters
        ----------
        forward: np.ndarray[float]
            Array of cumulated works for the forward transformation
        reverse: np.ndarray[float]
            Array of cumulated works for the reverse transformation


        Returns
        -------
        free_energies: np.ndarray[Float]
            List of bootstrapped free energies in units of kT.
        """
        import pymbar
        import numpy as np
        import numpy.typing as npt

        # Check to make sure forward and reverse work values match in length
        assert len(forward) == len(reverse), "Forward and reverse work values are not paired"

        all_dgs: npt.NDArray[float] = np.zeros(n_bootstraps)  # initialize dgs array

        traj_size = len(forward)
        for i in range(n_bootstraps):
            # Sample trajectory indices with replacement
            indices = np.random.choice(np.arange(traj_size), size=[traj_size], replace=True)
            dg, ddg = pymbar.bar.BAR(forward[indices], reverse[indices])
            all_dgs[i] = dg

        return all_dgs


class NonEquilibriumCyclingProtocol(Protocol):
    """
    Run RBFE calculations between two chemical states using alchemical NEQ cycling and `gufe` objects.

    Chemical states are assumed to have the same component keys, as in, stateA should be composed
    of the same type of components as components in stateB.
    """

    result_cls = NonEquilibriumCyclingProtocolResult

    def __init__(self, settings: Settings):
        super().__init__(settings)

    @classmethod
    def _default_settings(cls):
        from perses.protocols.settings import NonEquilibriumCyclingSettings
        return NonEquilibriumCyclingSettings(
                forcefield_settings=OpenMMSystemGeneratorFFSettings(),
                thermo_settings=ThermoSettings(temperature=300 * unit.kelvin),
                )

    # NOTE: create method should be really fast, since it would be running in the work units not the clients!!
    def _create(
            self,
            stateA: ChemicalSystem,
            stateB: ChemicalSystem,
            mapping: Optional[dict[str, ComponentMapping]] = None,
            extends: Optional[ProtocolDAGResult] = None,
    ) -> List[ProtocolUnit]:

        # Handle parameters
        if mapping is None:
            raise ValueError("`mapping` is required for this Protocol")
        if 'ligand' not in mapping:
            raise ValueError("'ligand' must be specified in `mapping` dict")
        if extends:
            raise NotImplementedError("Can't extend simulations yet")

        # inputs to `ProtocolUnit.__init__` should either be `Gufe` objects
        # or JSON-serializable objects
        num_replicates = self.settings.num_replicates

        simulations = [
            SimulationUnit(state_a=stateA, state_b=stateB, mapping=mapping, settings=self.settings, name=f"{replicate}")
            for replicate in range(num_replicates)]

        end = ResultUnit(name="result", simulations=simulations)

        return [*simulations, end]

    def _gather(
        self, protocol_dag_results: Iterable[ProtocolDAGResult]
    ) -> Dict[str, Any]:

        from collections import defaultdict
        outputs = defaultdict(list)
        for pdr in protocol_dag_results:
            for pur in pdr.protocol_unit_results:
                if pur.name == "result":
                    outputs["forward_work"].extend(pur.outputs["forward_work"])
                    outputs["reverse_work"].extend(pur.outputs["reverse_work"])
                    # TODO: below is broken; extending from a dict only puts in keys
                    # don't need it right now anyway
                    #outputs["work_file_paths"].extend(pur.outputs["paths"])

        # include the temperature so the ProtocolResult object can convert out
        # of kT units
        outputs['temperature'] = self.settings.thermo_settings.temperature

        # This can be populated however we want
        return outputs
