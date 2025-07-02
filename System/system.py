import numpy as np

class System:
    """A generalized control system

    Attributes
    ----------
    state_space : numpy.ndarray, optional
        The state of all signals for every timepoint
    name : str, optional
        The name of the system

    """
    def __init__(
            self, 
            initial_state_space,
            transition_functions,
            channel_names=None,
            output_channel_idx=None,
            name=None,
            verbose=False
            ):
        """Creates a system

        Parmeters
        ---------
        initial_state_space : 2D numpy.ndarray
            The initial state of every signal in the system. The shape is 2D,
            (number of channels, signal size). Where a signal in a channel is
            represented by a 1D vector of a user-specified length.
        transition_functions : list of None and tuples of (function, ndarray)
            The transition function leading to each signal. e.g. the
            transition function at index 0 would be creating the signal in
            channel 0. If the function is None then it is parsed as an input
            channel. When it is a tuple, it must be in the form 
            (callable_function, channel_mask), where channel_mask filters
            the channels being sent to the callable_function.
        channel_names : list of str, optional
            A list of channel names.
        output_channel_idx : 1D numpy.ndarray
            A index mask for the output channel
            NOTE: Multi-channel output not implmented yet
        name : str, optional
            The name of the system
        verbost : bool, optional
            flag to display intermediate steps

        """
        if len(initial_state_space) != len(transition_functions):
            raise ValueError(f"Length of state space \
                ({len(initial_state_space)}) does not match number of \
                transition functions ({len(transition_functions)})")
        self._initial_state_space = initial_state_space
        self._n_channels, self._signal_len = initial_state_space.shape
        self._transition_functions = transition_functions
        self._channel_names = channel_names
        self._subsystems = []
        self._input_channels_idxs = []
        for i, fn in enumerate(self._transition_functions):
            if fn is None:
                self._input_channels_idxs.append(i)
            elif isinstance(fn[0], System):
                self._subsystems.append(fn[0])
        self._output_channel_idx = output_channel_idx
        self._no_output_mask = np.ones(self._n_channels, dtype=bool)
        if self._output_channel_idx is not None:
            self._no_output_mask[output_channel_idx] = False
            if len(self._input_channels_idxs) > 0:
                self._no_output_mask[self._input_channels_idxs] = False            
        self.state_space = None
        self._current_step = None
        self.name = name
        self._verbose = verbose
        if self._verbose:
            print(f"input chan idxs: {self._input_channels_idxs}")


    def __call__(self, *input_channels):
        """Steps the state through their transition functions

        Parmeters
        ---------
        input_channels : list of 1D numpy.ndarray, optional
            The input signals of the system

        Returns
        -------
        1D numpy.ndarray, contingent
            If output_channel_idx is defined, returns the output channel
            of the system

        """
        if self._current_step is None:
            raise ValueError("There is no current step")
        _s = self._current_step

        # NOTE: attempted to code the following as parallel as possible
        # Transition all
        prev_step = self.state_space[_s-1]
        if len(self._input_channels_idxs) > 0:
            prev_step[self._input_channels_idxs] = np.array(input_channels)
        curr_step_list = [fn[0](*prev_step[fn[1]])
            if fn is not None else np.zeros(self._signal_len)
            for fn in self._transition_functions]
        # TODO: Add a demultiplexer incase theres one fn -> many channels
        curr_step = np.array(curr_step_list)

        # Set in matrix
        # NOTE: could use a short buffer instead...
        self.state_space[_s] = curr_step

        self._current_step += 1
        if self._output_channel_idx is not None:
            return curr_step[self._output_channel_idx]


    def go(self, number_of_steps=50):
        """Simulates the whole system for a set amount of timesteps

        Parmeters
        ---------
        input_channels : list of 1D numpy.ndarray, optional
            The input signals of the system

        Returns
        -------
        3D numpy.ndarray
            The deep state space of the system; a concatenation of the state 
            space of this system and the inner subsystems if it has them.
            The shape is 3D, (number of timepoints, total number of channels, 
            signal size). Where number of timepoints is the number of steps
            specified, total number of channels is the number channels and
            number of channels in composed subsystems.

        """
        self.deep_start(number_of_steps)
        # Loop per steps
        for step in range(1,number_of_steps):
            self()
        return self.deep_state_space()

    # ------------- #

    def deep_start(self, number_of_steps=50):
        # Starts the state spaces and the nested state spaces
        for subsys in self._subsystems:
            subsys.deep_start(number_of_steps=number_of_steps)
        # NOTE: populates entire history with the initial state space
        #   Could be an issue if the functions don't fully describe
        #   the state space and a channel is indeterminate for a timepoint,
        #   and/or is not overwritten at every timepoint.
        self.state_space = np.stack(
            [self._initial_state_space] * number_of_steps, axis=0) 
        self._current_step = 1

    def deep_state_space(self):
        # Retrieves the state spaces and the nested state spaces
        return np.concatenate([self.state_space[:, self._no_output_mask]] + 
            [subsys.deep_state_space() for subsys in self._subsystems],
            axis=1)

    def deep_channel_names(self):
        # Retrieves the channel names and the nested channel names
        sysname = f"{self.name} " if self.name is not None else ""
        if self._channel_names is None:
            valid_chans = np.arange(self._n_channels)[self._no_output_mask]
            outer_ch_names = [f"{sysname}ch_{i}" for i in valid_chans]
        else:
            outer_ch_names = [f"{sysname}{ch_n}" 
                for i, ch_n in enumerate(self._channel_names) 
                if self._no_output_mask[i]]
        return outer_ch_names + \
            [ch_name for subsys in self._subsystems
                for ch_name in subsys.deep_channel_names()]

# ========================================================================= #

class Agent(System):
    """A social agent control system

    """
    def __init__(
            self,
            sensor,
            comparator,
            effector,
            motor,
            feedback,
            signal_len,
            motor_prior=None,
            reference_initial=None,
            channel_names=None,
            **kwargs
            ):
        """Creates an agent

        Parmeters
        ---------
        sensor : (numpy.ndarray) -> numpy.ndarray
            Sensory compression function.
        comparator : (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
            Comparator function. 
        effector : (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
            Effector function.
        motor : (numpy.ndarray) -> numpy.ndarray
            Motor decompression function.
        feedback : (ndarray, ndarray, ndarray) -> ndarray
            Reference update function.
        signal_len : int
            Max length of a signal.
        motor_prior : 1D numpy.ndarray, optional
            The inital state of the effector signal.
        reference_initial : 1D numpy.ndarray, optional
            The inital state of the reference signal.
        channel_names : list of str, optional
            The names for each of the seven channels. Otherwise auto-assigned.

        """
        self._signal_len = signal_len
        n = signal_len

        if motor_prior is None:
            motor_prior = np.zeros((1,n))
        if reference_initial is None:
            reference_initial = np.zeros((1,n))
        state_space = np.concatenate(
            [np.zeros((1,n)),    # 0 input
             np.zeros((1,n)),    # 1 sense signal
             np.zeros((1,n)),    # 2 error signal
             motor_prior,        # 3 effect signal (motor state)
             np.zeros((1,n)),    # 4 motor signal / output
             reference_initial,  # 5 reference signal
             reference_initial]) # 6 reference initial

        transition_functions = [
            None,                # 0 input
            (sensor, [0]),       # 1 sense signal
            (comparator, [1,5]), # 2 error signal
            (effector, [2,3]),   # 3 effect signal
            (motor, [3]),        # 4 motor signal / output
            (feedback, [2,5,6]), # 5 reference signal
            (lambda x: x, [6])]  # 6 reference initial

        if channel_names is None:
            channel_names = ["input",
                "sense signal",
                "error signal",
                "effect signal (motor state)",
                "motor signal / output",
                "reference signal",
                "reference initial"]
        super().__init__(state_space,
            transition_functions,
            channel_names=channel_names,
            output_channel_idx=4,
            **kwargs)