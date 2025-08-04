import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import correlations_functions as fn

# Import the data to be plotted
data_og = pd.read_csv(r"E:/Computational_results/Databank_and_correlations_comparison/Databank/ICAO_data.csv", delimiter=";")
clmns = data_og.columns

# List of columns to keep
clmns = ["Pressure Ratio", "B/P Ratio", "Rated Thrust (kN)", "Ambient Baro Min (kPa)",
         "CO EI T/O (g/kg)",  "CO EI C/O (g/kg)", "CO EI App (g/kg)", "CO EI Idle (g/kg)",
         "HC EI T/O (g/kg)", "HC EI C/O (g/kg)", "HC EI App (g/kg)", "HC EI Idle (g/kg)",
         "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)",  "Fuel Flow App (kg/sec)", "Fuel Flow Idle (kg/sec)",
         "NOx EI T/O (g/kg)",  "NOx EI C/O (g/kg)",  "NOx EI App (g/kg)",  "NOx EI Idle (g/kg)", "NOx LTO Total mass (g)"]

# CF-56 and variants
cf56_range = [[61, 169]]
cf56_avg = fn.data_extraction(data_og, clmns, cf56_range)
 
cf56_avg["SFC T/O (g/kgkN)"] = cf56_avg["Fuel Flow T/O (kg/sec)"]/cf56_avg["Rated Thrust (kN)"]
cf56_avg["SFC Idle (g/kgkN)"] = cf56_avg["Fuel Flow Idle (kg/sec)"]/(0.07*cf56_avg["Rated Thrust (kN)"])
cf56_avg["SFC App (g/kgkN)"] = cf56_avg["Fuel Flow App (kg/sec)"]/(0.3*cf56_avg["Rated Thrust (kN)"])
cf56_avg["SFC C/O (g/kgkN)"] = cf56_avg["Fuel Flow C/O (kg/sec)"]/(0.85*cf56_avg["Rated Thrust (kN)"])

# V2500-E5
v2500_range = [[436,  462]]
v2500_avg = fn.data_extraction(data_og, clmns, v2500_range)

v2500_avg["SFC T/O (g/kgkN)"] = v2500_avg["Fuel Flow T/O (kg/sec)"]/v2500_avg["Rated Thrust (kN)"]
v2500_avg["SFC Idle (g/kgkN)"] = v2500_avg["Fuel Flow Idle (kg/sec)"]/(0.07*v2500_avg["Rated Thrust (kN)"])
v2500_avg["SFC App (g/kgkN)"] = v2500_avg["Fuel Flow App (kg/sec)"]/(0.3*v2500_avg["Rated Thrust (kN)"])
v2500_avg["SFC C/O (g/kgkN)"] = v2500_avg["Fuel Flow C/O (kg/sec)"]/(0.85*v2500_avg["Rated Thrust (kN)"])

# CFM Leap family
leap_range = [[169, 203]]
leap_avg = fn.data_extraction(data_og, clmns, leap_range)

leap_avg["SFC T/O (g/kgkN)"] = leap_avg["Fuel Flow T/O (kg/sec)"]/leap_avg["Rated Thrust (kN)"]
leap_avg["SFC Idle (g/kgkN)"] = leap_avg["Fuel Flow Idle (kg/sec)"]/(0.07*leap_avg["Rated Thrust (kN)"])
leap_avg["SFC App (g/kgkN)"] = leap_avg["Fuel Flow App (kg/sec)"]/(0.3*leap_avg["Rated Thrust (kN)"])
leap_avg["SFC C/O (g/kgkN)"] = leap_avg["Fuel Flow C/O (kg/sec)"]/(0.85*leap_avg["Rated Thrust (kN)"])

# PW1000 family
pw_range = [[526, 611]]
pw_avg = fn.data_extraction(data_og, clmns, pw_range)

pw_avg["SFC T/O (g/kgkN)"] = pw_avg["Fuel Flow T/O (kg/sec)"]/pw_avg["Rated Thrust (kN)"]
pw_avg["SFC Idle (g/kgkN)"] = pw_avg["Fuel Flow Idle (kg/sec)"]/(0.07*pw_avg["Rated Thrust (kN)"])
pw_avg["SFC App (g/kgkN)"] = pw_avg["Fuel Flow App (kg/sec)"]/(0.3*pw_avg["Rated Thrust (kN)"])
pw_avg["SFC C/O (g/kgkN)"] = pw_avg["Fuel Flow C/O (kg/sec)"]/(0.85*pw_avg["Rated Thrust (kN)"])

# Plot the results

# Flight phases
x_axis = ["Idle", "Take-off", "Climb-out", "Approach"]

# Figure No.1 - CO, NOx and UHC emissions indices as a function of the flight phase

fig1, ax1 = plt.subplots(1, 3, figsize = (12,6))
plt.subplots_adjust(left=0.1, right=0.9,
                    top=0.9, bottom=0.1,
                    wspace = 0.3)

ax1[0].set_xlabel("Flight Phases")
ax1[0].set_ylabel("NOx Emissions Index (g/kg)")
ax1[0].plot(x_axis, [cf56_avg["NOx EI Idle (g/kg)"], 
                     cf56_avg["NOx EI T/O (g/kg)"], 
                     cf56_avg["NOx EI C/O (g/kg)"], 
                     cf56_avg["NOx EI App (g/kg)"]],
            marker = 'o', label = "CFM56")
ax1[0].plot(x_axis, [v2500_avg["NOx EI Idle (g/kg)"], 
                     v2500_avg["NOx EI T/O (g/kg)"], 
                     v2500_avg["NOx EI C/O (g/kg)"], 
                     v2500_avg["NOx EI App (g/kg)"]],
            marker = 's', label = "V2500")
ax1[0].plot(x_axis, [leap_avg["NOx EI Idle (g/kg)"], 
                     leap_avg["NOx EI T/O (g/kg)"], 
                     leap_avg["NOx EI C/O (g/kg)"], 
                     leap_avg["NOx EI App (g/kg)"]],
            marker = '>', label = "LEAP")
ax1[0].plot(x_axis, [pw_avg["NOx EI Idle (g/kg)"], 
                     pw_avg["NOx EI T/O (g/kg)"], 
                     pw_avg["NOx EI C/O (g/kg)"], 
                     pw_avg["NOx EI App (g/kg)"]],
            marker = '*', label = "PW-1000")
ax1[0].grid(True)
ax1[0].legend()

ax1[1].set_xlabel("Flight Phases")
ax1[1].set_ylabel("CO Emissions Index (g/kg)")
ax1[1].plot(x_axis, [cf56_avg["CO EI Idle (g/kg)"], 
                     cf56_avg["CO EI T/O (g/kg)"], 
                     cf56_avg["CO EI C/O (g/kg)"], 
                     cf56_avg["CO EI App (g/kg)"]],
            marker = 'o', label = "CFM56")
ax1[1].plot(x_axis, [v2500_avg["CO EI Idle (g/kg)"], 
                     v2500_avg["CO EI T/O (g/kg)"], 
                     v2500_avg["CO EI C/O (g/kg)"], 
                     v2500_avg["CO EI App (g/kg)"]],
            marker = 's', label = "V2500")
ax1[1].plot(x_axis, [leap_avg["CO EI Idle (g/kg)"], 
                     leap_avg["CO EI T/O (g/kg)"], 
                     leap_avg["CO EI C/O (g/kg)"], 
                     leap_avg["CO EI App (g/kg)"]],
            marker = '>', label = "LEAP")
ax1[1].plot(x_axis, [pw_avg["CO EI Idle (g/kg)"], 
                     pw_avg["CO EI T/O (g/kg)"], 
                     pw_avg["CO EI C/O (g/kg)"], 
                     pw_avg["CO EI App (g/kg)"]],
            marker = '*', label = "PW-1000")
ax1[1].grid(True)
ax1[1].legend()

ax1[2].set_xlabel("Flight Phases")
ax1[2].set_ylabel("HC Emissions Index (g/kg)")
ax1[2].plot(x_axis, [cf56_avg["HC EI Idle (g/kg)"], 
                     cf56_avg["HC EI T/O (g/kg)"], 
                     cf56_avg["HC EI C/O (g/kg)"], 
                     cf56_avg["HC EI App (g/kg)"]],
            marker = 'o', label = "CFM56 family")
ax1[2].plot(x_axis, [v2500_avg["HC EI Idle (g/kg)"], 
                     v2500_avg["HC EI T/O (g/kg)"], 
                     v2500_avg["HC EI C/O (g/kg)"], 
                     v2500_avg["HC EI App (g/kg)"]],
            marker = 's', label = "V2500 family")
ax1[2].plot(x_axis, [leap_avg["HC EI Idle (g/kg)"], 
                     leap_avg["HC EI T/O (g/kg)"], 
                     leap_avg["HC EI C/O (g/kg)"], 
                     leap_avg["HC EI App (g/kg)"]],
            marker = '>', label = "LEAP family")
ax1[2].plot(x_axis, [pw_avg["HC EI Idle (g/kg)"], 
                     pw_avg["HC EI T/O (g/kg)"], 
                     pw_avg["HC EI C/O (g/kg)"], 
                     pw_avg["HC EI App (g/kg)"]],
            marker = '*', label = "PW-1000 family")
ax1[2].grid(True)
ax1[2].legend()

fig1.suptitle("Emissions Indices of various pollutants - ICAO emissions databank", fontsize=14)

# Figure No.2 - Averaged rated thrust and fuel flow for each flight phase
fig2, ax1 = plt.subplots(1,3, figsize = (12,6))
plt.subplots_adjust(left=0.1, right=0.9,
                    top=0.9, bottom=0.1,
                    wspace = 0.3)

ax1[0].set_xlabel("Flight phases")
ax1[0].set_ylabel("Averaged rated (maximum) thrust")
ax1[0].plot(x_axis, [0.07*cf56_avg["Rated Thrust (kN)"], 
                     cf56_avg["Rated Thrust (kN)"], 
                     0.85*cf56_avg["Rated Thrust (kN)"], 
                     0.3*cf56_avg["Rated Thrust (kN)"]],
            marker = 'o', label = "CFM56 family")
ax1[0].plot(x_axis, [0.07*v2500_avg["Rated Thrust (kN)"], 
                     v2500_avg["Rated Thrust (kN)"], 
                     0.85*v2500_avg["Rated Thrust (kN)"], 
                     0.3*v2500_avg["Rated Thrust (kN)"]],
            marker = 's', label = "V2500 family")
ax1[0].plot(x_axis, [0.07*leap_avg["Rated Thrust (kN)"], 
                     leap_avg["Rated Thrust (kN)"], 
                     0.85*leap_avg["Rated Thrust (kN)"], 
                     0.3*leap_avg["Rated Thrust (kN)"]],
            marker = '>', label = "LEAP family")
ax1[0].plot(x_axis, [0.07*pw_avg["Rated Thrust (kN)"], 
                     pw_avg["Rated Thrust (kN)"], 
                     0.85*pw_avg["Rated Thrust (kN)"], 
                     0.3*pw_avg["Rated Thrust (kN)"]],
            marker = '*', label = "PW-1000 family")
ax1[0].grid(True)
ax1[0].legend()

ax1[1].set_xlabel("Flight Phases")
ax1[1].set_ylabel("Fuel flow (kg/sec)")
ax1[1].plot(x_axis, [cf56_avg["Fuel Flow Idle (kg/sec)"], 
                     cf56_avg["Fuel Flow T/O (kg/sec)"], 
                     cf56_avg["Fuel Flow C/O (kg/sec)"], 
                     cf56_avg["Fuel Flow App (kg/sec)"]],
            marker = 'o', label = "CFM56")
ax1[1].plot(x_axis, [v2500_avg["Fuel Flow Idle (kg/sec)"], 
                     v2500_avg["Fuel Flow T/O (kg/sec)"], 
                     v2500_avg["Fuel Flow C/O (kg/sec)"], 
                     v2500_avg["Fuel Flow App (kg/sec)"]],
            marker = 's', label = "V2500")
ax1[1].plot(x_axis, [leap_avg["Fuel Flow Idle (kg/sec)"], 
                     leap_avg["Fuel Flow T/O (kg/sec)"], 
                     leap_avg["Fuel Flow C/O (kg/sec)"], 
                     leap_avg["Fuel Flow App (kg/sec)"]],
            marker = '>', label = "LEAP")
ax1[1].plot(x_axis, [pw_avg["Fuel Flow Idle (kg/sec)"], 
                     pw_avg["Fuel Flow T/O (kg/sec)"], 
                     pw_avg["Fuel Flow C/O (kg/sec)"], 
                     pw_avg["Fuel Flow App (kg/sec)"]],
            marker = '*', label = "PW-1000")
ax1[1].grid(True)
ax1[1].legend()

ax1[2].set_xlabel("Flight Phases")
ax1[2].set_ylabel("Specific Fuel Consumption (SFC, g/kgN)")
ax1[2].plot(x_axis, [1e3*cf56_avg["SFC Idle (g/kgkN)"], 
                     1e3*cf56_avg["SFC T/O (g/kgkN)"], 
                     1e3*cf56_avg["SFC C/O (g/kgkN)"], 
                     1e3*cf56_avg["SFC App (g/kgkN)"]],
            marker = 'o', label = "CFM56")
ax1[2].plot(x_axis, [1e3*v2500_avg["SFC Idle (g/kgkN)"], 
                     1e3*v2500_avg["SFC T/O (g/kgkN)"], 
                     1e3*v2500_avg["SFC C/O (g/kgkN)"], 
                     1e3*v2500_avg["SFC App (g/kgkN)"]],
            marker = 's', label = "V2500")
ax1[2].plot(x_axis, [1e3*leap_avg["SFC Idle (g/kgkN)"], 
                     1e3*leap_avg["SFC T/O (g/kgkN)"], 
                     1e3*leap_avg["SFC C/O (g/kgkN)"], 
                     1e3*leap_avg["SFC App (g/kgkN)"]],
            marker = '>', label = "LEAP")
ax1[2].plot(x_axis, [1e3*pw_avg["SFC Idle (g/kgkN)"], 
                     1e3*pw_avg["SFC T/O (g/kgkN)"], 
                     1e3*pw_avg["SFC C/O (g/kgkN)"], 
                     1e3*pw_avg["SFC App (g/kgkN)"]],
            marker = '*', label = "PW-1000")
ax1[2].grid(True)
ax1[2].legend()

fig2.suptitle("Averaged Thrust, Fuel flow and SFC values for various engines - ICAO emissions databank", fontsize=14)

plt.show()