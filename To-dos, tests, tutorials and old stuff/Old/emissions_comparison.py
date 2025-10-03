import pandas as pd
import numpy as np
import correlations_functions as fn
from matplotlib import pyplot as plt

## Extract data from csv file
data_og = pd.read_csv(r"E:/Computational_results/Databank_and_correlations_comparison/Databank/ICAO_data.csv", delimiter=";")

## List of columns to keep
clmns = ["Pressure Ratio", "B/P Ratio", "Rated Thrust (kN)", "Ambient Baro Min (kPa)",
         "CO EI T/O (g/kg)",  "CO EI C/O (g/kg)", "CO EI App (g/kg)", "CO EI Idle (g/kg)",
         "HC EI T/O (g/kg)", "HC EI C/O (g/kg)", "HC EI App (g/kg)", "HC EI Idle (g/kg)",
         "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)",  "Fuel Flow App (kg/sec)", "Fuel Flow Idle (kg/sec)",
         "NOx EI T/O (g/kg)",  "NOx EI C/O (g/kg)",  "NOx EI App (g/kg)",  "NOx EI Idle (g/kg)", "NOx LTO Total mass (g)"]

# CF-56 and variants
cf56_range = [[61, 169]]
cf56_avg = fn.data_extraction(data_og, clmns, cf56_range)

# V2500-E5
v2500_range = [[436,  462]]
v2500_avg = fn.data_extraction(data_og, clmns, v2500_range)

# CFM Leap family
leap_range = [[169, 203]]
leap_avg = fn.data_extraction(data_og, clmns, leap_range)

# PW1000 family
pw_range = [[526, 611]]
pw_avg = fn.data_extraction(data_og, clmns, pw_range)

# Engine parameters - CFM56
PRoverall = 32.7

## Data from correlation equations

# Dictionary definition: For every key: Tburner_inlet Tburner_outlet, Pburner, m_dot, FAR
dict = {
    "idle": [838.1, 1450, 3156.71, 23.02, 0.0139],
    "take-off": [837.9, 2250, 3152.59, 54.1, 0.0214],
    "climb-out": [846.27, 2100, 3240.46, 52.41, 0.0306],
    "approach": [828.5, 1400, 2909.36, 36.79, 0.01288]
}

# Extract the values for the pollutants
results = fn.data_processing(dict, PRoverall)
#print(f"ICAO Databank for NOx: {cf56_avg[14:19]}")
#print(f"Novelo: {results[:, 0]} gNOx/kgFuel")
#print(f"Lewis: {results[:, 1]} gNOx/kgFuel")
#print(f"Novelo: {results[:, 2]} gNOx/kgFuel")

## Plotting ##

# EINOx of various engines based on the databank
x_axis = ["Idle", "Take-off", "Climb-out", "Approach"]
cf56_EI = [cf56_avg[11], cf56_avg[8], cf56_avg[9], cf56_avg[10]]
v2500_EI = [v2500_avg[11], v2500_avg[8], v2500_avg[9], v2500_avg[10]]
leap_EI = [leap_avg[11], leap_avg[8], leap_avg[9], leap_avg[10]]
pw_EI = [pw_avg[11], pw_avg[8], pw_avg[9], pw_avg[10]]

# Fuel flow (FF) of various engines based on the databank
cf56_ff = [cf56_avg[7], cf56_avg[4], cf56_avg[5], cf56_avg[6]]
v2500_ff = [v2500_avg[7], v2500_avg[4], v2500_avg[5], v2500_avg[6]]
leap_ff = [leap_avg[7], leap_avg[4], leap_avg[5], leap_avg[6]]
pw_ff = [pw_avg[7], pw_avg[4], pw_avg[5], pw_avg[6]]

# Rated thrust of various engines based on the databank 
cf56_thrust = [0.07*cf56_avg[2], cf56_avg[2], 0.85*cf56_avg[2], 0.3*cf56_avg[2]]
v2500_thrust = [0.07*v2500_avg[2], v2500_avg[2], 0.85*v2500_avg[2], 0.3*v2500_avg[2]]
leap_thrust = [0.07*leap_avg[2], leap_avg[2], 0.85*leap_avg[2], 0.3*leap_avg[2]]
pw_thrust = [0.07*pw_avg[2], pw_avg[2], 0.85*pw_avg[2], 0.3*pw_avg[2]]

# Plot EIs and FF
fig0, ax1 = plt.subplots(1,2 ,figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.9,
                    top=0.9, bottom=0.1,
                    wspace = 0.3)

ax1[0].set_xlabel("Flight phases")
ax1[0].set_ylabel("NOx Emissions Index value (g/kg)")
ax1[0].plot(x_axis, cf56_EI, marker='o', label = "CFM56")
ax1[0].plot(x_axis, v2500_EI, marker = 's', label = "V2500")
ax1[0].plot(x_axis, leap_EI, marker = 'v', label = "Leap")
ax1[0].plot(x_axis, pw_EI, marker = '>', label = "PW1000")
ax1[0].grid(True)
ax1[0].legend()

ax1[1].set_xlabel("Flight phases")
ax1[1].set_ylabel("Fuel flow (kg/sec)")
ax1[1].plot(x_axis, cf56_ff, marker='o', label = "CFM56")
ax1[1].plot(x_axis, v2500_ff, marker = 's', label = "V2500")
ax1[1].plot(x_axis, leap_ff, marker = 'v', label = "Leap")
ax1[1].plot(x_axis, pw_ff, marker = '>', label = "PW1000")
ax1[1].grid(True)
ax1[1].legend()

fig0.suptitle("NOx emissions Index - Various engines - ICAO emissions databank", fontsize=14)
plt.savefig(r"E:/Computational_results/Plots/EIffdatabank.png")

# Thrust and Amount of pollutants produced
fig1, ax1 = plt.subplots(1,2 ,figsize=(10, 6))
plt.subplots_adjust(left=0.1, right=0.9,
                    top=0.9, bottom=0.1,
                    wspace = 0.3)

ax1[0].set_xlabel("Flight phases")
ax1[0].set_ylabel("Averaged rated thrust (kN)")
ax1[0].plot(x_axis, cf56_thrust, marker='o', label = "CFM56")
ax1[0].plot(x_axis, v2500_thrust, marker = 's', label = "V2500")
ax1[0].plot(x_axis, leap_thrust, marker = 'v', label = "Leap")
ax1[0].plot(x_axis, pw_thrust, marker = '>', label = "PW1000")
ax1[0].grid(True)
ax1[0].legend()

ax1[1].set_xlabel("Flight phases")
ax1[1].set_ylabel("Amount of NOx produced (g/s)")

res = [cf56_ff[i] * cf56_EI[i] for i in range(len(cf56_ff))]
ax1[1].plot(x_axis, res , marker='o', label = "CFM56")

res = [v2500_ff[i] * v2500_EI[i] for i in range(len(v2500_ff))]
ax1[1].plot(x_axis, res, marker = 's', label = "V2500")

res = [leap_ff[i] * leap_EI[i] for i in range(len(leap_ff))]
ax1[1].plot(x_axis, res, marker = 'v', label = "Leap")

res = [pw_ff[i] * pw_EI[i] for i in range(len(pw_ff))]
ax1[1].plot(x_axis, res, marker = '>', label = "PW1000")
ax1[1].grid(True)
ax1[1].legend()

fig1.suptitle("Averaged rated thrust and amount of NOx - Various engines - ICAO emissions databank", fontsize=14)
plt.savefig(r"E:/Computational_results/Plots/ThurstPollutantsDatabank.png")

# EINOx comparison of Correlation and Databank EIs
fig = plt.figure(figsize=(10,8))
plt.plot(x_axis, cf56_EI, label = "ICAO", marker = "*")#, color = "royalblue")
plt.plot(x_axis, results[:, 0], label = "Novelo et al.", marker = "o")#, color = "k")
#plt.plot(x_axis, results[:, 1], label = "Lewis", marker = "8")
plt.plot(x_axis, results[:, 2], label = "Rokke et. al", marker = "s")#, color = "green")
plt.plot(x_axis, results[:, 3], label = "Kyprianidis et. al", marker = "v")#, color = "red")
plt.title("EINOx value comparison - ICAO Databank and Correlation equations")
plt.xlabel("Flight phase")
plt.ylabel("Emissions Index (gNOx/kgFuel)")
plt.legend()
plt.grid()
plt.savefig(r"E:/Computational_results/Plots/Comparison.png")

# Plot the engine performance parameters per phase
# Extract parameter names
parameter_names = [
    "Burner Inlet Temp (K)",
    "Burner Outlet Temp (K)",
    "Burner Outlet Pressure (kPa)",
    "Mass Flow Rate (kg/s)",
    "Fuel-to-Air Ratio"
]

# Convert data into plottable format
phases = list(dict.keys())
values_by_parameter = list(zip(*dict.values()))

# First figure: Individual subplots for each of the first three parameters
fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
for i in range(3):
    axs1[i].plot(phases, values_by_parameter[i], marker='o')
    axs1[i].set_title(parameter_names[i])
    axs1[i].set_ylabel(parameter_names[i])
    axs1[i].grid(True)
fig1.suptitle("Burner Inlet/Outlet Temperatures and Outlet Pressure", fontsize=14)
plt.savefig(r"E:/Computational_results/Plots/Dict1.png")

# Second figure: Individual subplots for mass flow rate and fuel-to-air ratio
fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
for i in range(2):
    axs2[i].plot(phases, values_by_parameter[i + 3], marker='o')
    axs2[i].set_title(parameter_names[i + 3])
    axs2[i].set_ylabel(parameter_names[i + 3])
    axs2[i].grid(True)
fig2.suptitle("Mass Flow Rate and Fuel-to-Air Ratio", fontsize=14)
plt.savefig(r"E:/Computational_results/Plots/Dict2.png")

plt.show()





