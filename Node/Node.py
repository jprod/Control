from .Effector import Effector
import numpy as np


class Control_node:

    def __init__(self,
                 sensor,
                 comparator,
                 controller,
                 control_update,
                 behavioral_model=None,
                 system_estimate=None,
                 internal_model_update=None,
                 generate_reference=None,
                 ref_integration=None,
                 sen_integration=None,
                 parents=[],
                 children=[],
                 output_limits=(None, None)):
        self.sensor = sensor
        self.comparator = comparator
        self.controller = controller
        self.generate_reference = generate_reference
        self.control_update = control_update
        self.output_limits = output_limits
        self.parents = parents
        self.behavioral_model = behavioral_model,
        self.system_estimate = system_estimate,
        self.internal_model_update = internal_model_update,
        # past and current state
        self.previous_state = []
        self.sensory_signal = []
        # error is generally a contrast between target and sensed state
        self.error = []
        # target state
        self.reference = []
        # predicted state
        self.predicted_state = []
        # past and current behavioral outputs / motor commands
        self.previous_output = []
        self.output = []

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

    def set_reference(self, reference=[]):
        if not self.parents:
            self.reference = reference
            # self.reference = self.reference_update(self.reference, self.error)
        else:
            inputs = [p.get_output() for p in self.parents]
            self.reference = self.ref_integration(inputs, self.error)
        return self.reference

    def compare(self):
        if len(self.reference_signal) != len(self.sensory_signal):
            raise ValueError("Sensory signal must match reference signal.")
        self.error = self.comparator(
            reference_signal=self.reference, sensory_signal=self.sense, prediction=self.predicted_state)
        return self.error

    def control(self):
        # self.output = self.effector.get_output(error)
        if not self.previous_output:
            self.previous_output = np.ones(self.behavioral_model.ndim)
        output = self.controller(
            behavioral_model=self.behavioral_model, last_behavior=self.previous_output)
        self.previous_output = self.output
        self.output = output
        return self.output

    def internal_model(self):
        if not self.previous_state:
            self.previous_state = np.ones(self.behavioral_model.ndim)
        state_prediction = self.internal_model(
            estimate=self.system_estimate, last_state=self.previous_state, behavioral_model=self.behavioral_model, last_behavior=self.previous_behavior)
        return state_prediction

    def update_control(self):
        self.behavioral_model = self.control_update(
            error=self.error, behavioral_model=self.behavioral_model, last_behavior=self.previous_behavior)

    def go(self, observation, reference_signal=None):
        if reference_signal:
            self.set_reference(reference_signal)
        self.internal_model()
        self.sense(observation)
        self.compare()
        output = self.control()
        self.update_control()
        output = self.bound(output)
        self.output = output
        return output

    def get_output(self):
        return self.output

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
