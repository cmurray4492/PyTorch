# This is a series of functions collected from various books.

# 1D Data loader
import csv
from numpy import asarray

def load_data_1d(filename):
    """Load 1D Data"""
    with open(filename) as file:
        reader = csv.reader(file)
        header = next(reader)   # what does this line do?
        data = []
        for row in reader:
            data.append(row)
        data = asarray(data).astype(float)
    x = data[:, 0]  # input
    y = data[:, 1]  # output, target, grounf truth
    return (x, y)
