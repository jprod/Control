import numpy as np


class Control_node:

    def __init__(self,
                 sensor,
                 comparator,
                 controller,
                 control_update,
                 behavioral_model=None,
                 system_estimate=None,
                 internal_model=None,
                 internal_model_update=None,
                 reference_update=None,
                 reference=None,
                 init_behavior=None,
                 ref_integration=None,
                 sen_integration=None,
                 parents=[],
                 children=[],
                 output_limits=(None, None)):

        self.sensor = sensor
        self.comparator = comparator
        self.controller = controller
        self.reference_update = reference_update
        self.control_update = control_update
        self.output_limits = output_limits
        self.parents = parents
        self.behavioral_model = behavioral_model
        self.system_estimate = system_estimate
        self.internal_model = internal_model
        self.internal_model_update = internal_model_update
        # past and current state
        self.previous_state = []
        self.sensory_signal = []
        # error is generally a contrast between target and sensed state
        self.error = []
        # target state
        self.reference = []
        if reference is not None:
            self.reference = reference
        # predicted state
        self.predicted_state = []
        # past and current behavioral outputs / motor commands
        self.previous_output = []
        if init_behavior is not None:
            self.previous_output = init_behavior
        self.output = []
        # children/parent nodes
        self.children = children
        self.parents = parents

    def sense(self, observation=[]):
        if not self.children:
            # base sensor nodes get a signal passed directly to them as an observation
            self.previous_state = self.sensory_signal
            # self.sensory_signal = self.sen_integration(observation, self.error)
            self.sensory_signal = self.sensor(observation)
        else:
            inputs = [c.get_output() for c in self.children]
            self.previous_state = self.sensory_signal
            self.sensory_signal = self.sen_integration(inputs, self.error)

    def set_reference(self, reference=None, error=None):
        if not self.parents and reference and error:
            self.reference = self.reference_update(reference, error)
        elif not self.parents and error:
            self.reference = self.reference_update(self.reference, error)
        elif not self.parents and reference:
            self.reference = reference
        else:
            inputs = [p.get_output() for p in self.parents]
            self.reference = self.ref_integration(inputs, self.error)
        return self.reference

    def compare(self):
        if len(self.reference) != len(self.sensory_signal):
            raise ValueError("Sensory signal must match reference signal.")
        self.error = self.comparator(
            reference=self.reference, sensory_signal=self.sensory_signal, prediction=self.predicted_state)
        return self.error

    def control(self):
        # self.output = self.effector.get_output(error)

        if len(self.previous_output) == 0:
            self.previous_output = np.ones(self.behavioral_model.shape[0])
        output = self.controller(error=self.error,
                                 behavioral_model=self.behavioral_model, previous_output=self.previous_output)
        # if len(self.output) != 0:
        #    self.previous_output = self.output
        self.previous_output = output
        return output

    def generate_estimate(self):
        if len(self.previous_state) == 0:
            self.previous_state = np.ones(self.system_estimate.shape[0])
        if len(self.previous_output) == 0:
            self.previous_output = np.ones(self.behavioral_model.shape[0])
        self.predicted_state = self.internal_model(
            system_estimate=self.system_estimate, previous_state=self.previous_state, behavioral_model=self.behavioral_model, previous_output=self.previous_output)
        return self.predicted_state

    def update_control(self):
        self.behavioral_model = self.control_update(
            error=self.error, behavioral_model=self.behavioral_model, previous_output=self.previous_output)

    def go(self, observation, reference=None):
        if reference:
            self.set_reference(reference)
        self.generate_estimate()
        self.sense(observation)
        error = self.compare()
        self.set_reference(error=error)
        output = self.control()
        self.update_control()
        output = self.bound(output)
        self.output = output
        return output
    
    def get_reference(self):
        return self.reference

    def get_output(self):
        return self.previous_output

    def get_input(self):
        return self.sensory_signal

    def get_error(self):
        return self.error

    def bound(self, val):
        if self.output_limits[0] and self.output_limits[1]:
            lower = min(self.output_limits)
            upper = max(self.output_limits)
            if upper is not None and val > upper:
                return upper
            if lower is not None and val < lower:
                return lower
        return val
