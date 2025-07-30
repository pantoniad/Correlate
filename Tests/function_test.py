import pandas as pd
import numpy as np
import functions as fn

## Extract data from csv file
data_og = pd.read_csv(r"E:/Computational_results/Databank_and_correlations_comparison/Databank/ICAO_data.csv", delimiter=";")

## List of columns to keep
clmns = ["Pressure Ratio", "B/P Ratio", "Rated Thrust (kN)", "Ambient Baro Min (kPa)",
    "HC EI T/O (g/kg)", "HC EI C/O (g/kg)", "HC EI App (g/kg)", "HC EI Idle (g/kg)", "HC LTO Total mass (g)",
    "CO EI T/O (g/kg)", "CO EI C/O (g/kg)", "CO EI App (g/kg)", "CO EI Idle (g/kg)", "CO LTO Total Mass (g)",
    "NOx EI T/O (g/kg)",  "NOx EI C/O (g/kg)",  "NOx EI App (g/kg)",  "NOx EI Idle (g/kg)", "NOx LTO Total mass (g)"]

# Define starting and ending points of ranges. Take into consideration python's numbering scheme
data_range = [[109, 118], [153, 169]]

data_avg = fn.data_extraction(data_og, clmns, data_range)

# Test data_processing function

dict = {
    "idle": [838.1, 1450, 3156.71, 23.02, 0.0139],
    "take-off": [837.9, 2250, 3152.59, 54.1, 0.0214],
    "climb-out": [846.27, 2100, 3240.46, 52.41, 0.0306],
    "approach": [828.5, 1400, 2909.36, 36.79, 0.01288]
}

PRoverall = 32.7
results = fn.data_processing(dict, PRoverall)
print(f"Lewis: {results[:, 1]} ppmv")