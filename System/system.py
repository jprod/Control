import numpy as np

class System:
    def __init__(
            self, 
            initial_state_space,
            transition_functions,
            channel_names=None,
            output_channel_idx=None,
            name=None,
            verbose=False
            ):
        if len(initial_state_space) != len(transition_functions):
            raise ValueError("Length of state space does not match \
                              transition function")
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
        # Steps the state through their transition functions
        if self._current_step is None:
            raise ValueError("There is no current step")
        _s = self._current_step

        # NOTE: attempted to code the following as parallel as possible
        # Transition all
        curr_state = self.state_space[_s-1]
        if len(self._input_channels_idxs) > 0:
            curr_state[self._input_channels_idxs] = np.array(input_channels)
        next_step_list = [fn[0](*curr_state[fn[1]])
            if fn is not None else np.zeros(self._signal_len)
            for fn in self._transition_functions]
        # TODO: Add a demultiplexer incase theres one fn -> many channels
        next_step = np.array(next_step_list)

        # Set in matrix
        # NOTE: could use a short buffer instead...
        self.state_space[_s] = next_step

        self._current_step += 1
        if self._output_channel_idx is not None:
            return next_step[self._output_channel_idx]


    def go(self, number_of_steps=50):
        # Deep start state spaces
        self.deep_start(number_of_steps)
        # Loop per steps
        for step in range(1,number_of_steps):
            self()
        return self.deep_state_space()

    # ------------- #

    def deep_start(self, number_of_steps=50):
        # Starts the state spaces
        for subsys in self._subsystems:
            subsys.deep_start(number_of_steps=number_of_steps)
        self.state_space = np.stack(
            [self._initial_state_space] * number_of_steps, axis=0) 
        self._current_step = 1

    def deep_state_space(self):
        return np.concatenate([self.state_space[:, self._no_output_mask]] + 
            [subsys.deep_state_space() for subsys in self._subsystems],
            axis=1)

    def deep_channel_names(self):
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
    def __init__(
            self,
            sensor,
            comparator,
            effector,
            motor,
            feedback,
            signal_len,
            reference_initial=None,
            channel_names=None,
            **kwargs
            ):
        self._signal_len = signal_len
        n = signal_len
        self.motor_transform_matrix = np.eye(n)

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