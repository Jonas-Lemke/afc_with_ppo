"""
This file contains a class to handle the calibration of the mems sensors.

To get an correct wall sheer stress each voltage which is measured on the mems 
sensors gets subtracted by the voltage offset on the wind channel setup.
Than the corrected voltage is passed to function 
which maps an measured voltage to an wall sheer stress.
(This is done in sensors)
"""
import numpy.polynomial.polynomial as poly


class Calibration:
    def __init__(self):
        self.used_sensors = [109, 22, 61, 65, 94, 63, 19, 100]
        # self.used_sensors = [19]


        # coefs: dict {sensor number: list coefficients of polynomial}
        self.coefs = {}
        self.load_coefs()  # load data to coefs dict
        
        self.functions = {key: poly.Polynomial(coefs) for key, coefs in self.coefs.items()}

        self.offset_volt_channel = {109: 2.41542709e-02, 22: -0.01147612, 61: -6.79415448e-02, 65: 2.80349325e-02,
                                    94: -0.00204941, 63: 6.71327658e-02, 19: 0.00013941, 100: -0.00042395}

        # mean
        # offset: [2.47399132e-02 - 1.09672717e-02 - 6.74229956e-02  2.84354073e-02
        #          - 2.70351133e-03  6.74783316e-02  1.83017319e-04 - 7.79806615e-05]

    def load_coefs(self):
        """
        Loads the coefficients used to construct the polynomial functions (tau = f(volt).
        (The coefs in the loaded files originate from calculations performed outside this python project.)
        """
        for key in self.used_sensors:
            file_path = f'.\measurement\coefs\coefs_sensor_{key}.txt' # Improve code for cross platform use (Windows and Linux; now just works on Windows)
            with open(file_path, 'r') as file:
                line = file.readline().strip()
                self.coefs[key] = [float(value) for value in line.split('\t')]
