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

        # self.sensors.close_valve()
        self.sensors.open_valve()
        time.sleep(1.2)  # ~ update and write time

        self.sensors.set_pressure(volt=1)
        # self.sensors.set_pressure(volt=0.5)

    @staticmethod
    def calc_state(tau):
        """Helper function to turn measured tau into state"""

        # State: continuous (array = clipped tau)

        tau_copy = np.copy(tau)

        state = np.clip(tau_copy, a_min=-4, a_max=4)

        # state = np.round(state, decimals=3)

        # # # State: discrete
        # # conditions_for_state = [tau < -0.1, (tau >= -0.1) & (tau <= 0.1), tau > 0.1]
        # # values_for_state = [-1, 0, 1]
        #
        # conditions_for_state = [tau < -1.00, (tau >= -1.00) & (tau < -0.25), (tau >= -0.25) & (tau <= -0.05), (tau > -0.05) & (tau <= 0.05), (tau > 0.05) & (tau <= 0.25), (tau > 0.25) & (tau <= 1.00), tau > 1.00]
        # values_for_state = [-3, -2, -1, 0, 1, 2, 3]
        #
        # state = np.select(conditions_for_state, values_for_state)

        return state

    @staticmethod
    def calc_reward(tau):
        """Helper function to turn measured tau into reward"""

        tau_copy = np.copy(tau)

        conditions_for_reward = [tau_copy < 0.0, tau_copy >= 0.0]
        values_for_reward = [-1, 1]
        reward_array = np.select(conditions_for_reward, values_for_reward)


        # ### with weigths ###
        # # weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        # weights = np.array([82, 122, 162, 202, 242, 282, 386, 426])
        # # weights = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        #
        # weights_normalized = weights/np.sum(weights)
        # weighted_reward_array = reward_array * weights_normalized

        # ### w/o weights ###
        # weighted_reward_array = reward_array * [1/8]

        weighted_reward_array = reward_array

        ### reward ###
        reward = np.array(weighted_reward_array.sum()).reshape(1)
        
        return reward

    def reset(self):
        """Called to initiate a new episode. Resets the environment."""

        ### Wait until last pulse from last batch passed all sensors ###
        # time.sleep(0.2)  # 20 ms for pulse to pass sensors

        # self.sensors.close_valve()
        self.sensors.open_valve()
        time.sleep(0.2)

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
