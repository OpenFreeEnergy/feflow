# Adapted from perses: https://github.com/choderalab/perses/blob/protocol-neqcyc/perses/protocols/nonequilibrium_cycling.py

from typing import Optional, List, Dict, Any
from collections.abc import Iterable
from itertools import chain

import datetime
import logging
import pickle
import time

from gufe import SolventComponent, ProteinComponent
from gufe.settings import Settings
from gufe.chemicalsystem import ChemicalSystem
from gufe.mapping import ComponentMapping
from gufe.protocols import (
    Protocol,
    ProtocolUnit,
    ProtocolResult,
    ProtocolDAGResult,
)

from feflow.settings.small_molecules import OpenFFPartialChargeSettings

# TODO: Remove/change when things get migrated to openmmtools or feflow
from openfe.protocols.openmm_utils import system_creation
from openfe.protocols.openmm_rfe._rfe_utils.compute import get_openmm_platform

from openff.toolkit import Molecule as OFFMolecule
from openff.units import unit
from openff.units.openmm import to_openmm, from_openmm

from ..settings import NonEquilibriumCyclingSettings
from ..utils.data import serialize, deserialize
from ..utils.misc import (
    generate_omm_top_from_component,
    get_residue_index_from_atom_index,
    get_positions_from_component,
)

# Specific instance of logger for this module
logger = logging.getLogger(__name__)


class SetupUnit(ProtocolUnit):
    """
    Initial unit of the protocol. Sets up a Nonequilibrium cycling simulation given the chemical
    systems, mapping and settings.
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
            assert state_a.get("solvent") == state_b.get(
                "solvent"
            ), "Solvent parameters differ between solvent components."
        # check protein component is the same in both states if protein component is found
        if any(["protein" in state.components for state in (state_a, state_b)]):
            assert state_a.get("protein") == state_b.get(
                "protein"
            ), "Receptors in states are not compatible."

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
            "vacuum": ["ligand"],
        }
        for phase, keys in key_options.items():
            if all([key in state for state in states for key in keys]):
                detected_phase = phase
                break
        else:
            raise ValueError(
                "Could not detect phase from system states. Make sure the component in both systems match."
            )

        return detected_phase

    @staticmethod
    def _assign_openff_partial_charges(
        charge_settings: OpenFFPartialChargeSettings,
        off_small_mols: Iterable[OFFMolecule],
    ) -> None:
        """
        Assign partial charges to SmallMoleculeComponents given the specified settings,
        using the OpenFF toolkit.

        Parameters
        ----------
        charge_settings : OpenFFPartialChargeSettings
          Settings for controlling how the partial charges are assigned.
        off_small_mols : Iterable[OFFMolecule]
          Dictionary of OpenFF Molecules to add, keyed by
          state and SmallMoleculeComponent.
        """
        from feflow.utils.charge import assign_offmol_partial_charges

        for mol in off_small_mols:
            assign_offmol_partial_charges(
                offmol=mol,
                overwrite=False,
                method=charge_settings.partial_charge_method,
                toolkit_backend=charge_settings.off_toolkit_backend,
                generate_n_conformers=charge_settings.number_of_conformers,
                nagl_model=charge_settings.nagl_model,
            )

    def _execute(self, ctx, *, protocol, state_a, state_b, mapping, **inputs):
        """
        Execute the setup part of the nonequilibrium switching protocol.

        Parameters
        ----------
        ctx : gufe.protocols.protocolunit.Context
            The gufe context for the unit.
        protocol : gufe.protocols.Protocol
            The Protocol used to create this Unit. Contains key information
            such as the settings.
        state_a : gufe.ChemicalSystem
            The initial chemical system.
        state_b : gufe.ChemicalSystem
            The objective chemical system.
        mapping : gufe.mapping.LigandAtomMapping
            A dict featuring mappings between the two chemical systems.

        Returns
        -------
        dict : dict[str, str]
            Dictionary with paths to work arrays, both forward and reverse, and
            trajectory coordinates for systems A and B. As well as path for the
            pickled HTF object, mostly for debugging purposes.

        Notes
        -----
        * Here we assume the mapping is only between ``SmallMoleculeComponent``s.
        """
        # needed imports
        import openmm
        from openff.units.openmm import ensure_quantity
        from openmmtools.integrators import PeriodicNonequilibriumIntegrator
        from gufe.components import SmallMoleculeComponent
        from openfe.protocols.openmm_rfe import _rfe_utils
        from openfe.protocols.openmm_utils.system_validation import (
            get_alchemical_components,
        )
        from feflow.utils.hybrid_topology import HybridTopologyFactory
        from feflow.utils.charge import get_alchemical_charge_difference
        from feflow.utils.misc import (
            get_typed_components,
            register_ff_parameters_template,
        )

        # Check compatibility between states (same receptor and solvent)
        self._check_states_compatibility(state_a, state_b)

        phase = self._detect_phase(
            state_a, state_b
        )  # infer phase from systems and components

        # Get receptor components from systems if found (None otherwise)
        solvent_comp_a = get_typed_components(state_a, SolventComponent)
        protein_comps_a = get_typed_components(state_a, ProteinComponent)
        small_mols_a = get_typed_components(state_a, SmallMoleculeComponent)

        # Get alchemical components
        alchemical_comps = get_alchemical_components(state_a, state_b)

        # TODO: Do we need to change something in the settings? Does the Protein mutation protocol require specific settings?
        # Get all the relevant settings
        settings: NonEquilibriumCyclingSettings = protocol.settings
        # Get settings for system generator
        forcefield_settings = settings.forcefield_settings
        thermodynamic_settings = settings.thermo_settings
        integrator_settings = settings.integrator_settings
        charge_settings: OpenFFPartialChargeSettings = settings.partial_charge_settings
        solvation_settings = settings.solvation_settings
        alchemical_settings = settings.alchemical_settings

        # handle cache for system generator
        if settings.forcefield_cache is not None:
            ffcache = ctx.shared / settings.forcefield_cache
        else:
            ffcache = None

        system_generator = system_creation.get_system_generator(
            forcefield_settings=forcefield_settings,
            thermo_settings=thermodynamic_settings,
            integrator_settings=integrator_settings,
            cache=ffcache,
            has_solvent=bool(solvent_comp_a),
        )

        # Parameterizing small molecules
        self.logger.info("Parameterizing molecules")
        # Get small molecules from states
        # TODO: Refactor if/when gufe provides the functionality https://github.com/OpenFreeEnergy/gufe/issues/251
        state_a_small_mols = get_typed_components(state_a, SmallMoleculeComponent)
        state_b_small_mols = get_typed_components(state_b, SmallMoleculeComponent)
        all_small_mols = state_a_small_mols | state_b_small_mols

        # Generate and register FF parameters in the system generator template
        all_openff_mols = [comp.to_openff() for comp in all_small_mols]
        register_ff_parameters_template(
            system_generator, charge_settings, all_openff_mols
        )

        # c. get OpenMM Modeller + a dictionary of resids for each component
        state_a_modeller, _ = system_creation.get_omm_modeller(
            protein_comps=protein_comps_a,
            solvent_comp=solvent_comp_a,
            small_mols=small_mols_a,
            omm_forcefield=system_generator.forcefield,
            solvent_settings=solvation_settings,
        )

        # d. get topology & positions
        # Note: roundtrip positions to remove vec3 issues
        state_a_topology = state_a_modeller.getTopology()
        state_a_positions = to_openmm(from_openmm(state_a_modeller.getPositions()))

        # e. create the stateA System
        # Note: If there are no small mols ommffs requires a None
        state_a_system = system_generator.create_system(
            state_a_modeller.topology,
            molecules=(
                [mol.to_openff() for mol in state_a_small_mols]
                if state_a_small_mols
                else None
            ),
        )

        # 2. Get stateB system
        # a. Generate topology reusing state A topology as possible
        # Note: We are only dealing with single alchemical components
        state_b_alchem_top = generate_omm_top_from_component(
            alchemical_comps["stateB"][0]
        )
        state_b_alchem_pos = get_positions_from_component(alchemical_comps["stateB"][0])
        # We get the residue index from the mapping unique atom indices
        # NOTE: We assume single residue/point/component mutation here
        state_a_alchem_resindex = [
            get_residue_index_from_atom_index(
                state_a_topology, next(mapping.componentA_unique)
            )
        ]
        (
            state_b_topology,
            state_b_alchem_resids,
        ) = _rfe_utils.topologyhelpers.combined_topology(
            state_a_topology,
            state_b_alchem_top,
            exclude_resids=iter(state_a_alchem_resindex),
        )

        state_b_system = system_generator.create_system(
            state_b_topology,
            molecules=[mol.to_openff() for mol in state_b_small_mols],
        )

        # TODO: This doesn't have to be a ligand mapping. i.e. for protein mutation.
        # c. Define correspondence mappings between the two systems
        ligand_mappings = _rfe_utils.topologyhelpers.get_system_mappings(
            mapping.componentA_to_componentB,
            state_a_system,
            state_a_topology,
            state_a_alchem_resindex,
            state_b_system,
            state_b_topology,
            state_b_alchem_resids,
            # These are non-optional settings for this method
            fix_constraints=True,
        )

        # Handle charge corrections/transformations
        # Get the change difference between the end states
        # and check if the charge correction used is appropriate
        charge_difference = get_alchemical_charge_difference(
            mapping,
            forcefield_settings.nonbonded_method,
            alchemical_settings.explicit_charge_correction,
            # TODO: I don't understand why this isn't erroring when it's vacuum leg. review
            solvent_comp_a,  # Solvent comp in a is expected to be the same as in b
        )

        if alchemical_settings.explicit_charge_correction:
            alchem_water_resids = _rfe_utils.topologyhelpers.get_alchemical_waters(
                state_a_topology,
                state_a_positions,
                charge_difference,
                alchemical_settings.explicit_charge_correction_cutoff,
            )
            _rfe_utils.topologyhelpers.handle_alchemical_waters(
                alchem_water_resids,
                state_b_topology,
                state_b_system,
                ligand_mappings,
                charge_difference,
                solvent_comp_a,
            )

        # d. Finally get the positions
        state_b_positions = _rfe_utils.topologyhelpers.set_and_check_new_positions(
            ligand_mappings,
            state_a_topology,
            state_b_topology,
            old_positions=ensure_quantity(state_a_positions, "openmm"),
            insert_positions=state_b_alchem_pos,
        )

        # TODO: handle the literals directly in the HTF object (issue #42)
        # Get softcore potential settings
        if alchemical_settings.softcore_LJ.lower() == "gapsys":
            softcore_LJ_v2 = True
        elif alchemical_settings.softcore_LJ.lower() == "beutler":
            softcore_LJ_v2 = False
        # TODO: We need to test HTF for protein mutation cases, probably.
        #  What are ways to quickly check an HTF is correct?
        # Now we can create the HTF from the previous objects
        hybrid_factory = HybridTopologyFactory(
            state_a_system,
            state_a_positions,
            state_a_topology,
            state_b_system,
            state_b_positions,
            state_b_topology,
            old_to_new_atom_map=ligand_mappings["old_to_new_atom_map"],
            old_to_new_core_atom_map=ligand_mappings["old_to_new_core_atom_map"],
            use_dispersion_correction=alchemical_settings.use_dispersion_correction,
            softcore_alpha=alchemical_settings.softcore_alpha,
            softcore_LJ_v2=softcore_LJ_v2,
            softcore_LJ_v2_alpha=alchemical_settings.softcore_alpha,
            interpolate_old_and_new_14s=alchemical_settings.turn_off_core_unique_exceptions,
        )
        ####### END OF SETUP #########

        system = hybrid_factory.hybrid_system
        positions = hybrid_factory.hybrid_positions

        # Set up integrator
        temperature = to_openmm(thermodynamic_settings.temperature)
        integrator_settings = settings.integrator_settings
        integrator = PeriodicNonequilibriumIntegrator(
            alchemical_functions=settings.lambda_functions,
            nsteps_neq=integrator_settings.nonequilibrium_steps,
            nsteps_eq=integrator_settings.equilibrium_steps,
            splitting=integrator_settings.splitting,
            timestep=to_openmm(integrator_settings.timestep),  # needs openmm Quantity
            temperature=temperature,
        )

        # Set up context
        platform = get_openmm_platform(settings.engine_settings.compute_platform)
        context = openmm.Context(system, integrator, platform)
        context.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
        context.setPositions(positions)

        try:
            # Minimize
            openmm.LocalEnergyMinimizer.minimize(context)

            # SERIALIZE SYSTEM, STATE, INTEGRATOR
            # need to set velocities to temperature so serialized state features velocities,
            # which is important for usability by the Folding@Home openmm-core
            thermodynamic_settings = settings.thermo_settings
            temperature = to_openmm(thermodynamic_settings.temperature)
            context.setVelocitiesToTemperature(temperature)

            # state needs to include positions, forces, velocities, and energy
            # to be usable by the Folding@Home openmm-core
            state_ = context.getState(
                getPositions=True, getForces=True, getVelocities=True, getEnergy=True
            )
            system_ = context.getSystem()
            integrator_ = context.getIntegrator()

            htf_outfile = ctx.shared / "hybrid_topology_factory.pickle"
            system_outfile = ctx.shared / "system.xml.bz2"
            state_outfile = ctx.shared / "state.xml.bz2"
            integrator_outfile = ctx.shared / "integrator.xml.bz2"

            # Serialize HTF, system, state and integrator
            with open(htf_outfile, "wb") as htf_file:
                pickle.dump(hybrid_factory, htf_file)
            serialize(system_, system_outfile)
            serialize(state_, state_outfile)
            serialize(integrator_, integrator_outfile)

        finally:
            # Explicit cleanup for GPU resources
            del context, integrator

        return {
            "system": system_outfile,
            "state": state_outfile,
            "integrator": integrator_outfile,
            "phase": phase,
            "initial_atom_indices": hybrid_factory.initial_atom_indices,
            "final_atom_indices": hybrid_factory.final_atom_indices,
            "topology_path": htf_outfile,
        }


class CycleUnit(ProtocolUnit):
    """
    Monolithic unit for the cycle part of the simulation.
    It runs a number of NEq cycles from the outputs of a setup unit and stores the work computed in
    numpy-formatted files, to be analyzed by a result unit.
    """

    @staticmethod
    def extract_positions(context, initial_atom_indices, final_atom_indices):
        """
        Extract positions from initial and final systems based from the hybrid topology.

        Parameters
        ----------
        context: openmm.Context
            Current simulation context where from extract positions.
        hybrid_topology_factory: HybridTopologyFactory
            Hybrid topology factory where to extract positions and mapping information

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
        import numpy as np

        # Get positions from current openmm context
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)

        # Get indices for initial and final topologies in hybrid topology
        initial_indices = np.asarray(initial_atom_indices)
        final_indices = np.asarray(final_atom_indices)

        initial_positions = positions[initial_indices, :]
        final_positions = positions[final_indices, :]

        return initial_positions, final_positions

    def _execute(self, ctx, *, protocol, setup, **inputs):
        """
        Execute the simulation part of the Nonequilibrium switching protocol using GUFE objects.

        Parameters
        ----------
        ctx : gufe.protocols.protocolunit.Context
            The gufe context for the unit.
        protocol : gufe.protocols.Protocol
            The Protocol used to create this Unit. Contains key information
            such as the settings.
        setup : gufe.protocols.ProtocolUnit
            The SetupUnit

        Returns
        -------
        dict : dict[str, str]
            Dictionary with paths to work arrays, both forward and reverse, and trajectory coordinates for systems
            A and B.
        """
        import numpy as np
        import openmm
        import openmm.unit as openmm_unit
        from openmmtools.integrators import PeriodicNonequilibriumIntegrator

        # Setting up logging to file in shared filesystem
        file_logger = logging.getLogger("neq-cycling")
        output_log_path = ctx.shared / "feflow-neq-cycling.log"
        file_handler = logging.FileHandler(output_log_path, mode="w")
        file_handler.setLevel(logging.DEBUG)  # TODO: Set to INFO in production
        log_formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(log_formatter)
        file_logger.addHandler(file_handler)

        # Get state, system, and integrator from setup unit
        system = deserialize(setup.outputs["system"])
        state = deserialize(setup.outputs["state"])
        integrator = deserialize(setup.outputs["integrator"])
        PeriodicNonequilibriumIntegrator.restore_interface(integrator)

        # Get atom indices for either end of the hybrid topology
        initial_atom_indices = setup.outputs["initial_atom_indices"]
        final_atom_indices = setup.outputs["final_atom_indices"]

        # Extract settings from the Protocol
        settings = protocol.settings

        # Set up context
        platform = get_openmm_platform(settings.engine_settings.compute_platform)
        context = openmm.Context(system, integrator, platform)
        context.setState(state)

        # Equilibrate
        thermodynamic_settings = settings.thermo_settings
        temperature = to_openmm(thermodynamic_settings.temperature)
        context.setVelocitiesToTemperature(temperature)

        # Extract settings used below
        neq_steps = settings.integrator_settings.nonequilibrium_steps
        eq_steps = settings.integrator_settings.equilibrium_steps
        traj_save_frequency = settings.traj_save_frequency
        work_save_frequency = (
            settings.work_save_frequency
        )  # Note: this is divisor of traj save freq.
        selection_expression = settings.atom_selection_expression

        try:
            # Prepare objects to store positions -- empty lists so far
            (
                forward_eq_initial,
                forward_eq_final,
                forward_neq_initial,
                forward_neq_final,
            ) = (
                [],
                [],
                [],
                [],
            )
            (
                reverse_eq_final,
                reverse_eq_initial,
                reverse_neq_initial,
                reverse_neq_final,
            ) = (
                [],
                [],
                [],
                [],
            )

            # Coarse number of steps -- each coarse consists of work_save_frequency steps
            coarse_eq_steps = int(
                eq_steps / work_save_frequency
            )  # Note: eq_steps is multiple of work save steps
            coarse_neq_steps = int(
                neq_steps / work_save_frequency
            )  # Note: neq_steps is multiple of work save steps

            # TODO: Also get the GPU information (plain try-except with nvidia-smi)

            # Equilibrium (lambda = 0)
            # start timer
            start_time = time.perf_counter()
            for step in range(coarse_eq_steps):
                integrator.step(work_save_frequency)
                file_logger.debug(
                    f"coarse step: {step}: saving work (freq {work_save_frequency})"
                )
                # Save positions
                if step % traj_save_frequency == 0:
                    file_logger.debug(
                        f"coarse step: {step}: saving trajectory (freq {traj_save_frequency})"
                    )
                    initial_positions, final_positions = self.extract_positions(
                        context, initial_atom_indices, final_atom_indices
                    )
                    forward_eq_initial.append(initial_positions)
                    forward_eq_final.append(final_positions)
            # Make sure trajectories are stored at the end of the eq loop
            file_logger.debug(
                f"coarse step: {step}: saving trajectory (freq {traj_save_frequency})"
            )
            initial_positions, final_positions = self.extract_positions(
                context, initial_atom_indices, final_atom_indices
            )
            forward_eq_initial.append(initial_positions)
            forward_eq_final.append(final_positions)

            eq_forward_time = time.perf_counter()
            eq_forward_walltime = datetime.timedelta(
                seconds=eq_forward_time - start_time
            )
            file_logger.info(
                f"replicate_{self.name} Forward (lamba = 0) equilibration time : {eq_forward_walltime}"
            )

            # Run neq
            # Forward (0 -> 1)
            # Initialize works with current value
            forward_works = [integrator.get_protocol_work(dimensionless=True)]
            for fwd_step in range(coarse_neq_steps):
                integrator.step(work_save_frequency)
                forward_works.append(integrator.get_protocol_work(dimensionless=True))
                if fwd_step % traj_save_frequency == 0:
                    initial_positions, final_positions = self.extract_positions(
                        context, initial_atom_indices, final_atom_indices
                    )
                    forward_neq_initial.append(initial_positions)
                    forward_neq_final.append(final_positions)
            # Make sure trajectories are stored at the end of the neq loop
            initial_positions, final_positions = self.extract_positions(
                context, initial_atom_indices, final_atom_indices
            )
            forward_neq_initial.append(initial_positions)
            forward_neq_final.append(final_positions)

            neq_forward_time = time.perf_counter()
            neq_forward_walltime = datetime.timedelta(
                seconds=neq_forward_time - eq_forward_time
            )
            file_logger.info(
                f"replicate_{self.name} Forward nonequilibrium time (lambda 0 -> 1): {neq_forward_walltime}"
            )

            # Equilibrium (lambda = 1)
            for step in range(coarse_eq_steps):
                integrator.step(work_save_frequency)
                if step % traj_save_frequency == 0:
                    initial_positions, final_positions = self.extract_positions(
                        context, initial_atom_indices, final_atom_indices
                    )
                    reverse_eq_initial.append(
                        initial_positions
                    )  # TODO: Maybe better naming not old/new but initial/final
                    reverse_eq_final.append(final_positions)
            # Make sure trajectories are stored at the end of the eq loop
            initial_positions, final_positions = self.extract_positions(
                context, initial_atom_indices, final_atom_indices
            )
            reverse_eq_initial.append(initial_positions)
            reverse_eq_final.append(final_positions)

            eq_reverse_time = time.perf_counter()
            eq_reverse_walltime = datetime.timedelta(
                seconds=eq_reverse_time - neq_forward_time
            )
            file_logger.info(
                f"replicate_{self.name} Reverse (lambda 1) time: {eq_reverse_walltime}"
            )

            # Reverse work (1 -> 0)
            reverse_works = [integrator.get_protocol_work(dimensionless=True)]
            for rev_step in range(coarse_neq_steps):
                integrator.step(work_save_frequency)
                reverse_works.append(integrator.get_protocol_work(dimensionless=True))
                if rev_step % traj_save_frequency == 0:
                    initial_positions, final_positions = self.extract_positions(
                        context, initial_atom_indices, final_atom_indices
                    )
                    reverse_neq_initial.append(initial_positions)
                    reverse_neq_final.append(final_positions)
            # Make sure trajectories are stored at the end of the neq loop
            initial_positions, final_positions = self.extract_positions(
                context, initial_atom_indices, final_atom_indices
            )
            forward_eq_initial.append(initial_positions)
            forward_eq_final.append(final_positions)

            neq_reverse_time = time.perf_counter()
            neq_reverse_walltime = datetime.timedelta(
                seconds=neq_reverse_time - eq_reverse_time
            )
            file_logger.info(
                f"replicate_{self.name} Reverse nonequilibrium time (lambda 1 -> 0): {neq_reverse_walltime}"
            )
            # Elapsed time for the whole cycle
            cycle_walltime = datetime.timedelta(seconds=neq_reverse_time - start_time)
            file_logger.info(
                f"replicate_{self.name} Nonequilibrium cycle total walltime: {cycle_walltime}"
            )

            # Computing performance in ns/day
            timestep = to_openmm(settings.integrator_settings.timestep)
            simulation_time = 2 * (eq_steps + neq_steps) * timestep
            walltime_in_seconds = cycle_walltime.total_seconds() * openmm_unit.seconds
            estimated_performance = simulation_time.value_in_unit(
                openmm_unit.nanosecond
            ) / walltime_in_seconds.value_in_unit(
                openmm_unit.days
            )  # in ns/day
            file_logger.info(
                f"replicate_{self.name} Estimated performance: {estimated_performance} ns/day"
            )

            # Serialize works
            phase = setup.outputs["phase"]
            forward_work_path = ctx.shared / f"forward_{phase}_{self.name}.npy"
            reverse_work_path = ctx.shared / f"reverse_{phase}_{self.name}.npy"
            with open(forward_work_path, "wb") as out_file:
                np.save(out_file, forward_works)
            with open(reverse_work_path, "wb") as out_file:
                np.save(out_file, reverse_works)

            # Store timings and performance info in dictionary
            timing_info = {
                "eq_forward_time_in_s": eq_forward_walltime.total_seconds(),
                "neq_forward_time_in_s": neq_forward_walltime.total_seconds(),
                "eq_reverse_time_in_s": eq_reverse_walltime.total_seconds(),
                "neq_reverse_time_in_s": neq_reverse_walltime.total_seconds(),
                "performance_in_ns_day": estimated_performance,
            }

            # TODO: Do we need to save the trajectories?
            # Serialize trajectories
            forward_eq_old_path = ctx.shared / f"forward_eq_old_{phase}_{self.name}.npy"
            forward_eq_new_path = ctx.shared / f"forward_eq_new_{phase}_{self.name}.npy"
            forward_neq_old_path = (
                ctx.shared / f"forward_neq_old_{phase}_{self.name}.npy"
            )
            forward_neq_new_path = (
                ctx.shared / f"forward_neq_new_{phase}_{self.name}.npy"
            )
            reverse_eq_new_path = ctx.shared / f"reverse_eq_new_{phase}_{self.name}.npy"
            reverse_eq_old_path = ctx.shared / f"reverse_eq_old_{phase}_{self.name}.npy"
            reverse_neq_old_path = (
                ctx.shared / f"reverse_neq_old_{phase}_{self.name}.npy"
            )
            reverse_neq_new_path = (
                ctx.shared / f"reverse_neq_new_{phase}_{self.name}.npy"
            )

            with open(forward_eq_old_path, "wb") as out_file:
                np.save(out_file, np.array(forward_eq_initial))
            with open(forward_eq_new_path, "wb") as out_file:
                np.save(out_file, np.array(forward_eq_final))
            with open(reverse_eq_old_path, "wb") as out_file:
                np.save(out_file, np.array(reverse_eq_initial))
            with open(reverse_eq_new_path, "wb") as out_file:
                np.save(out_file, np.array(reverse_eq_final))
            with open(forward_neq_old_path, "wb") as out_file:
                np.save(out_file, np.array(forward_neq_initial))
            with open(forward_neq_new_path, "wb") as out_file:
                np.save(out_file, np.array(forward_neq_final))
            with open(reverse_neq_old_path, "wb") as out_file:
                np.save(out_file, np.array(reverse_neq_initial))
            with open(reverse_neq_new_path, "wb") as out_file:
                np.save(out_file, np.array(reverse_neq_final))

            # Saving trajectory paths in dictionary structure
            trajectory_paths = {
                "forward_eq_initial": forward_eq_old_path,
                "forward_eq_final": forward_eq_new_path,
                "forward_neq_initial": forward_neq_old_path,
                "forward_neq_final": forward_neq_new_path,
                "reverse_eq_initial": reverse_eq_old_path,
                "reverse_eq_final": reverse_eq_new_path,
                "reverse_neq_initial": reverse_neq_old_path,
                "reverse_neq_final": reverse_neq_new_path,
            }
        finally:
            # Explicit cleanup for GPU resources
            del context, integrator

        return {
            "forward_work": forward_work_path,
            "reverse_work": reverse_work_path,
            "trajectory_paths": trajectory_paths,
            "log": output_log_path,
            "timing_info": timing_info,
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
            forward_work = np.load(simulation.outputs["forward_work"])
            reverse_work = np.load(simulation.outputs["reverse_work"])

            forward_works.append(forward_work - forward_work[0])
            reverse_works.append(reverse_work - reverse_work[0])

        # Path dictionary
        paths_dict = {
            "forward_work": [
                simulation.outputs["forward_work"] for simulation in simulations
            ],
            "reverse_work": [
                simulation.outputs["reverse_work"] for simulation in simulations
            ],
        }

        return {
            "forward_work": forward_works,
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
        free_energy: unit.Quantity
            Free energy estimate in units of kcal/mol.

        """
        import numpy as np
        import numpy.typing as npt
        import pymbar

        forward_work = [i[-1] for i in self.data["forward_work"]]
        reverse_work = [i[-1] for i in self.data["reverse_work"]]

        forward_work: npt.NDArray[float] = np.array(forward_work)
        reverse_work: npt.NDArray[float] = np.array(reverse_work)
        free_energy, error = pymbar.bar.BAR(forward_work, reverse_work)

        return (
            free_energy * unit.k * self.data["temperature"] * unit.avogadro_constant
        ).to("kcal/mol")

    def get_uncertainty(self, n_bootstraps=1000):
        """
        Estimate uncertainty using standard deviation of the distribution of bootstrapped
        free energy (dg) samples.

        Parameters
        ----------
        n_bootstraps

        Returns
        -------
        free_energy_uncertainty: unit.Quantity
            Uncertainty on the free energy estimate in units of kcal/mol.

        """
        import numpy as np
        import numpy.typing as npt

        forward_work = [i[-1] for i in self.data["forward_work"]]
        reverse_work = [i[-1] for i in self.data["reverse_work"]]

        forward: npt.NDArray[float] = np.array(forward_work)
        reverse: npt.NDArray[float] = np.array(reverse_work)

        all_dgs = self._do_bootstrap(forward, reverse, n_bootstraps=n_bootstraps)

        # TODO: Check if standard deviation is a good uncertainty estimator
        return (
            np.std(all_dgs) * unit.k * self.data["temperature"] * unit.avogadro_constant
        ).to("kcal/mol")

    def get_rate_of_convergence(self): ...

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
        assert len(forward) == len(
            reverse
        ), "Forward and reverse work values are not paired"

        all_dgs: npt.NDArray[float] = np.zeros(n_bootstraps)  # initialize dgs array

        traj_size = len(forward)
        for i in range(n_bootstraps):
            # Sample trajectory indices with replacement
            indices = np.random.choice(
                np.arange(traj_size), size=[traj_size], replace=True
            )
            dg, ddg = pymbar.bar.BAR(forward[indices], reverse[indices])
            all_dgs[i] = dg

        return all_dgs


class NonEquilibriumCyclingProtocol(Protocol):
    """
    Run RBFE calculations between two chemical states using alchemical NEQ cycling and `gufe` objects.

    Chemical states are assumed to have the same component keys, as in, stateA should be composed
    of the same type of components as components in stateB.
    """

    _simulation_unit = CycleUnit
    result_cls = NonEquilibriumCyclingProtocolResult

    def __init__(self, settings: Settings):
        super().__init__(settings)

    @classmethod
    def _default_settings(cls):
        from feflow.settings import (
            NonEquilibriumCyclingSettings,
            PeriodicNonequilibriumIntegratorSettings,
        )
        from gufe.settings import OpenMMSystemGeneratorFFSettings, ThermoSettings
        from openfe.protocols.openmm_utils.omm_settings import (
            OpenMMSolvationSettings,
            OpenMMEngineSettings,
        )
        from openfe.protocols.openmm_rfe.equil_rfe_settings import AlchemicalSettings

        return NonEquilibriumCyclingSettings(
            forcefield_settings=OpenMMSystemGeneratorFFSettings(),
            thermo_settings=ThermoSettings(
                temperature=300 * unit.kelvin, pressure=1 * unit.bar
            ),
            solvation_settings=OpenMMSolvationSettings(),
            partial_charge_settings=OpenFFPartialChargeSettings(),
            alchemical_settings=AlchemicalSettings(softcore_LJ="gapsys"),
            integrator_settings=PeriodicNonequilibriumIntegratorSettings(),
            engine_settings=OpenMMEngineSettings(),
        )

    # NOTE: create method should be really fast, since it would be running in the work units not the clients!!
    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: Optional[dict[str, ComponentMapping]] = None,
        extends: Optional[ProtocolDAGResult] = None,
    ) -> list[ProtocolUnit]:
        # Handle parameters
        if mapping is None:
            raise ValueError("`mapping` is required for this Protocol")
        if extends:
            raise NotImplementedError("Can't extend simulations yet")

        # inputs to `ProtocolUnit.__init__` should either be `Gufe` objects
        # or JSON-serializable objects
        num_cycles = self.settings.num_cycles

        setup = SetupUnit(
            protocol=self,
            state_a=stateA,
            state_b=stateB,
            mapping=mapping,
            name="setup",
        )

        simulations = [
            self._simulation_unit(protocol=self, setup=setup, name=f"{replicate}")
            for replicate in range(num_cycles)
        ]

        end = ResultUnit(name="result", simulations=simulations)

        return [*simulations, end]

    def _gather(
        self, protocol_dag_results: Iterable[ProtocolDAGResult]
    ) -> dict[str, Any]:
        from collections import defaultdict

        outputs = defaultdict(list)
        for pdr in protocol_dag_results:
            for pur in pdr.protocol_unit_results:
                if pur.name == "result":
                    outputs["forward_work"].extend(pur.outputs["forward_work"])
                    outputs["reverse_work"].extend(pur.outputs["reverse_work"])
                    # TODO: below is broken; extending from a dict only puts in keys
                    # don't need it right now anyway
                    # outputs["work_file_paths"].extend(pur.outputs["paths"])

        # include the temperature so the ProtocolResult object can convert out
        # of kT units
        outputs["temperature"] = self.settings.thermo_settings.temperature

        # This can be populated however we want
        return outputs
