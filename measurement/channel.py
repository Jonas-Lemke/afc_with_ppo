"""
This file contains the Channels used by the InputOutputManager.
"""

import nidaqmx
import numpy as np
from nidaqmx.constants import RegenerationMode


class ChannelVoltageIn:
    """
    Class for a Single or Multiple Channel Object with an input task to measure voltage.
    """
    def __init__(self, channel: str, sampling_rate: int = None, num_samples: int = None, min_val: float = None, max_val: float = None,
                 terminal_config=nidaqmx.constants.TerminalConfiguration.DEFAULT):
        """
        Initializes ChannelVoltageIn.

        Parameters
        ----------
        channel: str
            Name of one or multiple channels e.g. "Dev1/ai0" or "Dev1/ai0, Dev1/ai1".
        sampling_rate: int
            Specifies the sampling rate in samples per second.
        num_samples: int
            Specifies the number of samples to acquire for the channel.
        min_val: float
            Specifies in units the minimum value you expect to measure.
        max_val: float
            Specifies in units the maximum value you expect to measure.
        terminal_config: nidaqmx.constants.TerminalConfiguration
            Specifies the input terminal configuration for the channel.
        """

        # Assign variables
        self.channel = channel
        self.sampling_rate = sampling_rate
        self.num_samples = num_samples  # measuring_time * sampling_rate
        self.terminal_config = terminal_config

        # Set up task
        self.task = nidaqmx.Task()

        # Add an analog input channel
        self.task.ai_channels.add_ai_voltage_chan(self.channel, min_val=min_val, max_val=max_val, terminal_config=terminal_config)

        # Start task
        self.task.start()

    def read_single_voltage(self) -> list:
        """
        Function reads in one single voltage.

        Returns
        -------
        volt: list[float]
            for multiple channels returns list of floats
            for single channel returns single float
        """
        volt = self.task.read()
        return volt

    def close_task(self):
        """Function is just to make further code more readable. Tasks should be closed at the end."""
        self.task.stop()
        self.task.close()


class ChannelVoltageOut:
    """
    Class for a Single of Multi Channel Object with an output task to generate voltage.
    """
    def __init__(self, channel: str, sampling_rate: int, num_of_channels: int):
        """
        Initializes ChannelVoltageOut.

        Parameters
        ----------
        channel: str
            Name of the channel e.g. "Dev1/ao0".
        sampling_rate: int
            Specifies the sampling rate in samples per channel per second.
        num_of_channels: int
            Specifies the number of channels of the output task.
        """

        # Assign variables
        self.channel = channel
        self.sampling_rate = sampling_rate
        self.num_of_channels = num_of_channels

        # Set up task
        self.task = nidaqmx.Task()

        # Add an analog output channel
        self.task.ao_channels.add_ao_voltage_chan(self.channel)

        # Configure regeneration mode
        # after buffer is written same data is written again in a loop until there is new data
        self.task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

        # Start task
        self.task.start()

    def write_voltage_output(self, voltage: float):
        """Starts the task with the inserted voltage."""
        # The NI USB-6281 needs at least a buffer with two identical samples

        # single channel
        if self.num_of_channels == 1:
            self.task.write(np.full(2, voltage))
        # multi channel
        else:
            self.task.write(np.vstack([np.full(2, voltage)] * self.num_of_channels))

    def close_task(self):
        """Function to make further code more readable. Tasks should be closed at the end."""
        self.task.stop()
        self.task.close()
