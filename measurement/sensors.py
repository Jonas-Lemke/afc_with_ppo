"""
This file contains the Class InputOutputManager used for I/O and setup of the NI USB-6281 sensors.
"""

import numpy as np
import nidaqmx

import measurement.channel as channel
import measurement.calibration as calibration


class InputOutputManager:
    """Class for I/O of the NI USB-6281"""
    def __init__(self, output_channel, input_channel):
        """
        Initializes Sensors.

        Parameters
        ----------
        output_channel: list[str]
            list with strings of the output channel names
        input_channel: list[str]
            list with strings of input channel names
        """

        ### Output channel names ###
        self.pressure_channel = output_channel[0]  # first output for reservoir ("Dev1/ao0")
        self.valve_channel = output_channel[1]  # second output for pressure jet actuators ("Dev1/ao1")

        ### Output tasks ###
        self.pressure_output_task = channel.ChannelVoltageOut(channel=self.pressure_channel, sampling_rate=1000,
                                                              num_of_channels=1)
        self.valve_output_task = channel.ChannelVoltageOut(channel=self.valve_channel, sampling_rate=1000,
                                                           num_of_channels=1)
        ### Input channel name ###
        self.input_channel = ", ".join(input_channel)  # ai_channels need a comma separated str instead of a list

        ### Input task ###
        self.input_task = channel.ChannelVoltageIn(channel=self.input_channel,
                                                   terminal_config=nidaqmx.constants.TerminalConfiguration.RSE,
                                                   min_val=-10, max_val=10)
        
        ### Sensor Calibration Class ###
        self.calib = calibration.Calibration()

    @staticmethod
    def vol_flow_cal(volt):
        """Helper function to convert measured voltage (V) into a volume flow (l/min)"""
        # For a volume flow sensor with the range 2-200 l/min
        volt_min = 0
        volt_max = 10
        vol_flow_min = 2
        vol_flow_max = 200

        # Get linear function
        p = np.polynomial.polynomial.polyfit((volt_min, volt_max), (vol_flow_min, vol_flow_max), 1)
        vol_flow = p[0] + volt * p[1]

        return vol_flow
    
    def measure_vol_flow(self):
        """Measures current volume flow (l/min)"""
        volt = self.input_task.read_single_voltage()
        volt_vol_flow = volt[0]  # ai0
        vol_flow = self.vol_flow_cal(volt_vol_flow)
        return vol_flow

    def tau_cal(self, volt_measured):
        """Helper function to convert  measured voltage (V) into wall shear stress"""
        list_tau = []
        for num_sensor, volt_sensor in zip(self.calib.used_sensors, volt_measured):
            f = self.calib.functions[num_sensor]
            offset = self.calib.offset_volt_channel[num_sensor]
            tau = f(volt_sensor - offset)
            list_tau.append(tau)
        return list_tau

    def measure_tau(self):
        """Measures current wall shear stress on all mems sensors"""
        volt = self.input_task.read_single_voltage()
        volt_tau = volt[1:]  # ai1 - ai8
        tau = self.tau_cal(volt_tau)
        return tau

    def measure_vol_flow_and_tau(self):
        """Measures current volume flow and wall shear stress"""
        # measure voltage
        volt = self.input_task.read_single_voltage()
        # separate voltage array
        volt_vol_flow = volt[0]  # ai0
        volt_tau = volt[1:]  # ai1 - ai8
        # calculate volume flow and wall shear stress
        vol_flow = self.vol_flow_cal(volt_vol_flow)
        tau = self.tau_cal(volt_tau)
        return vol_flow, tau

    def measure_voltage(self):
        """Function for debugging"""
        volt = self.input_task.read_single_voltage()
        volt = volt[1:]
        return volt

    def measure_voltage_corrected(self):
        """Function for debugging"""
        volt = self.input_task.read_single_voltage()
        volt = volt[1:]
        corrections = [self.calib.offset_volt_channel[sensor] for sensor in self.calib.used_sensors]
        corr_volt = [volt[i] - corrections[i] for i in range(len(volt))]
        return corr_volt

    def set_pressure(self, volt):
        """Set pressure on reservoir"""
        self.pressure_output_task.write_voltage_output(voltage=volt)

    def open_valve(self):
        """Start voltage output of 5 volt on valve channel (PJAs)"""
        self.valve_output_task.write_voltage_output(voltage=5)

    def close_valve(self):
        """Stop voltage output on valve channel (PJAs)"""
        self.valve_output_task.write_voltage_output(voltage=0.0)

    def vol_flow_regulator(self):
        # TO-DO: write this function which uses vol_flow_output_task to regulate volume flow
        raise NotImplementedError

    def close_tasks(self):
        """Closes all tasks. Call this function after all measurements are done"""
        self.valve_output_task.close_task()
        self.pressure_output_task.close_task()
        self.input_task.close_task()
