import numpy as np


class Control_node:

    def __init__(self,
                 sensor,
                 comparator,
                 effector,
                 motor,
                 system_estimate=None,
                 internal_model=None,
                 internal_model_update=None,
                 reference_update=None,
                 reference=None,
                 motor_prior=None,
                 motor_transform=None,
                 init_behavior=None,
                 ref_integration=None,
                 sen_integration=None,
                 parents=[],
                 children=[],
                 output_limits=(None, None)):

        self.sensor = sensor
        self.comparator = comparator
        self.effector = effector
        self.motor = motor
        self.reference_update = reference_update
        self.behavior_limits = output_limits
        self.parents = parents
        self.system_estimate = system_estimate
        self.internal_model = internal_model
        self.internal_model_update = internal_model_update
        # past and current state
        self.previous_sense = []
        self.sensory_signal = []
        # error is generally a contrast between target and sensed state
        self.error = []
        # motor state
        self.motor_state = []
        if motor_prior is not None:
            self.motor_state = motor_prior
        self.motor_transform = motor_transform
        # target state
        self.reference = []
        if reference is not None:
            self.reference = reference
        # predicted state
        self.predicted_state = []
        # past and current behavioral outputs / motor commands
        self.previous_move = []
        if init_behavior is not None:
            self.previous_move = init_behavior
        # self.behavior = [] # UNUSED RN
        # children/parent nodes
        self.children = children
        self.parents = parents

    def sense(self, observation=[]):
        if not self.children:
            # base sensor nodes get a signal passed directly to them as an observation
            self.previous_sense = self.sensory_signal
            # self.sensory_signal = self.sen_integration(observation, self.error)
            self.sensory_signal = self.sensor(observation)
        else:
            inputs = [c.get_output() for c in self.children]
            self.previous_sense = self.sensory_signal
            self.sensory_signal = self.sen_integration(inputs, self.error)
        return self.sensory_signal

    def set_reference(self, reference=None, error=None):
        # init case
        if reference is not None and error is None:
            self.reference = reference
            return self.reference
        # update case
        if self.reference_update:
            if error is not None:
                if reference is not None:
                    self.reference = self.reference_update(reference=reference, error=error)
                elif self.reference is None:
                    raise ValueError("no reference value in node")
                else:
                    self.reference = self.reference_update(reference=self.reference, error=error)
        # no reference update function
        #else:
            #if error:
                #raise ValueError("no reference update function given")
        if reference is None and self.reference is None:
            raise ValueError("no reference value provided")
        
        return self.reference
    
        # if not self.parents and reference and error:
        #     self.reference = self.reference_update(reference, error)
        # elif not self.parents and error and self.reference_update:
        #     self.reference = self.reference_update(self.reference, error)
        # elif not self.parents and reference:
        #     self.reference = reference
        # elif not self.parents and error:
        #     self.reference = self.reference_update(error)
        # elif reference:
        #     self.reference = reference
        #     #inputs = [p.get_output() for p in self.parents]
        #     #self.reference = self.ref_integration(inputs, self.error)
        # return self.reference

    def compare(self, sensory_signal):
        if len(self.reference) != len(sensory_signal):
            raise ValueError("Sensory signal must match reference signal.")
        self.error = self.comparator(
            reference=self.reference, sensory_signal=self.sensory_signal, prediction=self.predicted_state)
        return self.error

    def effect(self, error):
        motor_signal = self.effector(self.motor_state, error, self.motor_transform)
        self.motor_state = motor_signal
        return motor_signal

    def _prev_move_init(self):
        if len(self.previous_move) == 0:
            self.previous_move = np.ones(self.behavioral_model.shape[0])
    
    def move(self, motor_signal):
        self._prev_move_init()
        behavior = self.motor(motor_signal=motor_signal)
        # if len(self.behavior) != 0:
        #    self.previous_move = self.behavior
        self.previous_move = behavior
        return behavior

    def generate_estimate(self):
        if len(self.previous_sense) == 0:
            self.previous_sense = np.ones(self.system_estimate.shape[0])
        self._prev_move_init()
        self.predicted_state = self.internal_model(
            system_estimate=self.system_estimate, 
            previous_sense=self.previous_sense,
            previous_move=self.previous_move)
        return self.predicted_state

    # def update_control(self):
    #     self.behavioral_model = self.control_update(
    #         error=self.error, behavioral_model=self.behavioral_model, previous_move=self.previous_move)

    # MAIN
    def go(self, observation, reference=None):
        if reference:
            self.set_reference(reference=reference)
        self.generate_estimate()

        #sensory compression
        sense_signal = self.sense(observation)

        #comparison and refrence signal adjustment
        error = self.compare(sensory_signal=sense_signal)
        self.set_reference(error=error)

        #effector
        motor_signal = self.effect(error=error)

        #motor decompression and bounding
        behavior = self.move(motor_signal=motor_signal)
        # self.update_move()
        behavior = self.bound(behavior)
        
        # self.behavior = behavior
        return behavior
    
    def get_reference(self):
        return self.reference

    def get_output(self):
        return self.previous_move

    def get_input(self):
        return self.sensory_signal

    def get_error(self):
        return self.error

    def bound(self, val):
        if self.behavior_limits[0] and self.behavior_limits[1]:
            lower = min(self.behavior_limits)
            upper = max(self.behavior_limits)
            if upper is not None and val > upper:
                return upper
            if lower is not None and val < lower:
                return lower
        return val
