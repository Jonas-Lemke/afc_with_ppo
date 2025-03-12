"""
This file contains general-purpose functions
"""

import os
import csv


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def create_file(file_path, header_list):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(header_list)  


def write_file(file_path, data_list):
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(data_list)
