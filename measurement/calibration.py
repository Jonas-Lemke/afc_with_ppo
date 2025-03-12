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
        # coefs: dict {sensor number: list coefficients of polynomial}
        self.used_sensors = [49, 22, 61, 65, 94, 100, 101, 117]
        # self.used_sensors = [49, 22, 61, 65, 94, 100, 101]

        self.coefs = {}
        self.load_coefs()  # load data to coefs dict
        
        self.functions = {key: poly.Polynomial(coefs) for key, coefs in self.coefs.items()}
        
        # self.offset_volt_channel = {49: 2.6869826292084125e-05, 22: 5.431068586008668e-05, 61: -0.06895357169030487, 
        #                             65: 0.028204988364619276, 94: -2.6023145871408575e-05, 100: -4.7369510243398816e-05, 
        #                             101: 0.00013795188141973959, 117: 0.004238526646693181}
        
        self.offset_volt_channel = {49: -0.0003514113161353088, 22: 0.001169278667600178, 61: -0.06818321640671017,
                                    65: 0.028888466396779836, 94: 0.0017063926316046008, 100: 0.0001039243497398688,
                                    101: 0.00020045993582982433, 117: 0.004113633980806188}

        # Mittlere Spannungswerte bei An- und Ausgeschalteten Sensoren:
        # Aus: [-0.004456502345245983, -0.005218230083104015, -0.004031506892135421, -0.00608353939103015, -0.003100467135290407, -0.0043469890032215, -0.004009078380135778, -0.014047520863459985]
        # An:  [-0.0002248240892427274, 0.0011126943695879258, -0.0684167839559072, 0.028720733513711037, 0.001635002950626833, -0.0002496734057035088, 0.00021314911896941124, 0.0035142043220699535]

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
