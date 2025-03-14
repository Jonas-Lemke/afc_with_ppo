"""
This file contains the custom wind tunnel environment for the PPO.
"""

import time
import numpy as np
import measurement.sensors as sm


class WindTunnelEnv:
    """Class for custom wind tunnel environment"""
    def __init__(self, num_states, num_actions, output_channel, input_channel):
        """Initialize wind tunnel environment"""
        self.num_states = num_states  # Number of state inputs
        self.num_actions = num_actions  # Number of possibles output actions
        self.output_channel = output_channel
        self.input_channel = input_channel

        self.sensors = sm.InputOutputManager(output_channel=self.output_channel, input_channel=self.input_channel)  # Initialize sensors

        self.sensors.close_valve()
        self.sensors.set_pressure(volt=1)

    @staticmethod
    def calc_state(tau):
        """Helper function to turn measured tau into state"""

        # State: continuous (array = clipped tau)
        state = np.clip(tau, a_min=-5, a_max=5)

        # # State: discrete (array = pos. tau = 1, tau ~ 0 = 0 , neg. tau = -1)
        # # conditions_for_state = [tau < -0.3, (tau >= -0.3) & (tau <= 0.3), tau > 0.3]
        # # values_for_state = [-1, 0, 1]
        # conditions_for_state = [tau < 0.0, tau > 0.0]
        # values_for_state = [-1, 1]
        # state = np.select(conditions_for_state, values_for_state)
        return state

    @staticmethod
    def calc_reward(tau):
        """Helper function to turn measured tau into reward"""

        conditions_for_reward = [tau < -0.3, (tau >= -0.3) & (tau <= 0.3), tau > 0.3]
        values_for_reward = [-1, 0, 1]
        reward_array = np.select(conditions_for_reward, values_for_reward)

        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        weights_normalized = weights/np.sum(weights)
        weighted_reward_array = reward_array * weights_normalized

        # weighted_reward_array = reward_array * [1/7]  # -> keine gewichtung

        reward = np.array(weighted_reward_array.sum()).reshape(1)
        return reward

    def reset(self):
        """Called to initiate a new episode. Resets the environment."""

        self.sensors.close_valve()
        # self.sensors.set_pressure(volt=2)  # prob. not needed
        
        ### Wait until last pulse from last batch passed all sensors ###
        time.sleep(0.025)  # 20 ms for pulse to pass sensors

        ### Measure volume flow and tau ###
        vol_flow, measured_tau = self.sensors.measure_vol_flow_and_tau()

        ### Change tau to work with the rest of the code ###
        tau = np.array(measured_tau).reshape(1, self.num_states)

        ### Turn measured tau into state ###
        state = self.calc_state(tau)

        return state, vol_flow, measured_tau

    def step(self, action):
        """Computes the state of the environment after applying an action"""

        ### Get a real copy for reward calculation ###
        # old_state = np.copy(self.state)

        ### Apply action ###
        if action:
            self.sensors.open_valve()
        else:
            self.sensors.close_valve()

        # ### Measure tau ###
        # tau = np.array(self.sensors.measure_tau()).reshape(1, self.num_states)

        ### Measure volume flow and tau ###
        vol_flow, measured_tau = self.sensors.measure_vol_flow_and_tau()

        # Change tau to work with the rest of the code
        tau = np.array(measured_tau).reshape(1, self.num_states)

        ### Turn measured tau into state ###
        state = self.calc_state(tau)

        ### Turn measured tau into reward ###
        reward = self.calc_reward(tau)

        ### Check terminal state ###
        terminated = False  # Terminal state not used here

        return state, reward, terminated, vol_flow, measured_tau

    def end_measurement(self):
        self.sensors.close_valve()
        self.sensors.close_tasks()
