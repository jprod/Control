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
        transition_functions : list of None, ints, tuples of (function, ndarray)
            The transition function leading to each signal. e.g. the
            transition function at index 0 would be creating the signal in
            channel 0. If the function is None then it is parsed as an input
            channel. When the function is an int, it is interpreted as the 
            channel for the target of a demux. You cannot link a demuxed 
            channel to another demuxed channel. When it is a tuple, 
            it must be in the form (callable_function, channel_mask), 
            where channel_mask filters the channels being sent to the 
            callable_function.
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
        self._demux_channels = []
        for i, fn in enumerate(self._transition_functions):
            if fn is None:
                self._input_channels_idxs.append(i)
            elif isinstance(fn, (int, np.integer)):
                self._demux_channels.append(fn) # src
            elif isinstance(fn[0], System):
                self._subsystems.append(fn[0])
        self._demux_channels = np.unique(self._demux_channels)
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
        prev_step = self.state_space[_s-1]
        # Input
        if len(self._input_channels_idxs) > 0:
            prev_step[self._input_channels_idxs] = np.array(input_channels)
        # Transition all
        curr_step_list_undemux = [
            np.zeros(self._signal_len) if fn is None else
            fn if isinstance(fn, (int, np.integer)) else 
            fn[0](*prev_step[fn[1]])
            for fn in self._transition_functions]
        # Demux
        if len(self._demux_channels) >= 1:
            for i in self._demux_channels:
                curr_step_list_undemux[i] = iter(curr_step_list_undemux[i])
            curr_step_list = [
                s if isinstance(s, np.ndarray) else 
                next(curr_step_list_undemux[s]) 
                    if isinstance(s, (int, np.integer)) else 
                next(s) for s in curr_step_list_undemux]
        else:
            curr_step_list = curr_step_list_undemux
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
            [ss for ss in 
            [subsys.deep_state_space() for subsys in self._subsystems]
            if ss is not None],
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
                for ch_name in subsys.deep_channel_names() 
                if ch_name is not None]

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

# ========================================================================= #

class Lag(System):
    """A queued buffer that selects random values for a jittered lag

    """
    def __init__(
            self,
            signal_len,
            lags_per_n=None,
            max_lag=10,
            buffer_fill=0.0,
            name="lag",
            invis=True,
            random_mask=False,
            ):
        self._signal_len = signal_len
        if lags_per_n is not None:
            self.max_lag = np.max(lags_per_n)
            self.lag_mask = lags_per_n
        elif max_lag == 0:
            self.max_lag = max_lag
            self.lag_mask = np.zeros(self._signal_len,int)
        else:
            self.max_lag = max_lag
            self.lag_mask = np.random.randint(0,self.max_lag+1,self._signal_len)
        self._signal_buffer_init = np.full((max_lag+1, signal_len), buffer_fill)
        self.signal_buffer = None
        self._initial_state_space = np.full((1,signal_len), buffer_fill)
        self.state_space = None
        self._current_step = None
        self._invis = invis
        self.name = name
        self.random_mask = random_mask
        

    def __call__(self, input_channel):
        if self.signal_buffer is None:
            raise ValueError("There is no current buffer")
        # Input
        self.signal_buffer = np.roll(self.signal_buffer,1,axis=0)
        self.signal_buffer[0,:] = input_channel
        if self.random_mask:
            self.lag_mask = np.random.randint(0,self.max_lag+1,self._signal_len)
        if not self._invis:
            _s = self._current_step
            self.state_space[_s] = input_channel
            self._current_step += 1
        return self.signal_buffer[self.lag_mask].diagonal()
        
    def deep_start(self, number_of_steps):
        self.signal_buffer = self._signal_buffer_init
        if not self._invis:
            self.state_space = np.stack(
                [self._initial_state_space] * number_of_steps, axis=0) 
            self._current_step = 1

    def deep_state_space(self):
        if not self._invis:
            return self.state_space
        return None

    def deep_channel_names(self):
        if not self._invis:
            return [self.name]
        return [None]