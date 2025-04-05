import mdtraj


class BaseSwitchingUnit(ProtocolUnit):
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

    def _execute(self, ctx, *, protocol, md_unit, index, **inputs):
        """
        Execute the simulation part of the Nonequilibrium switching protocol using GUFE objects.

        Parameters
        ----------
        ctx : gufe.protocols.protocolunit.Context
            The gufe context for the unit.
        protocol : gufe.protocols.Protocol
            The Protocol used to create this Unit. Contains key information
            such as the settings.
        md_unit : gufe.protocols.ProtocolUnit
            The SetupUnit
        index: int
            TODO: Index for the snapshot to use as input

        Returns
        -------
        dict : dict[str, str]
            Dictionary with paths to work arrays, both forward and reverse, and trajectory coordinates for systems
            A and B.
        """
        import openmm
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

        system = deserialize(md_unit.inputs["setup"].outputs["system"])
        state = deserialize(md_unit.inputs["setup"].outputs["state"])
        integrator = deserialize(md_unit.inputs["setup"].outputs["integrator"])

        PeriodicNonequilibriumIntegrator.restore_interface(integrator)

        # Get atom indices for either end of the hybrid topology
        initial_atom_indices = setup.outputs["initial_atom_indices"]
        final_atom_indices = setup.outputs["final_atom_indices"]

        # Extract settings from the Protocol
        settings = protocol.settings

        # Load positions from snapshots
        xtc_file = md_unit.outputs["production_trajectory"]
        md_traj_ob = mdtraj.load_frame(xtc_file, index=index)
        input_positions = md_traj_ob.openmm_positions(0)
        # Set up context
        platform = get_openmm_platform(settings.engine_settings.compute_platform)
        context = openmm.Context(system, integrator, platform)
        context.setState(state)
        # TODO: This is kinda ugly, is there a better way to set positions?
        context.setPositions(input_positions)

        # Setting velocities to temperatures
        thermodynamic_settings = settings.thermo_settings
        temperature = to_openmm(thermodynamic_settings.temperature)
        context.setVelocitiesToTemperature(temperature)

        # Extract settings used below
        neq_steps = settings.integrator_settings.nonequilibrium_steps
        traj_save_frequency = settings.traj_save_frequency
        work_save_frequency = (
            settings.work_save_frequency
        )  # Note: this is divisor of traj save freq.
        selection_expression = settings.atom_selection_expression

        try:
            # Coarse number of steps -- each coarse consists of work_save_frequency steps
            coarse_neq_steps = int(
                neq_steps / work_save_frequency
            )  # Note: neq_steps is multiple of work save steps

            # TODO: Also get the GPU information (plain try-except with nvidia-smi)


            integrator.step(NSTEPS)




            # Equilibrium (lambda = 0)
            # start timer
            start_time = time.perf_counter()
            # Run neq
            # Forward (0 -> 1)
            # Initialize works with current value
            forward_works = []
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

            # TODO: We should return the work in one direction
