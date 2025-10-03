import pandas as pd
import numpy as np
import Classes.FuelFlow_class as ffms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import data
df = pd.read_csv(r"E:/Correlate/Databank/ICAO_data.csv", delimiter=";")

# Define and get only the collumns needed
clmns = ["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow Idle (kg/sec)", 
         "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)","Fuel Flow App (kg/sec)",
         "NOx EI Idle (g/kg)", "NOx EI T/O (g/kg)", "NOx EI C/O (g/kg)",
         "NOx EI App (g/kg)"]

data = df[clmns]
print(data.head())

# Define and get only the rows for the needed engine
engRange = [[61, 169]]
cfm56 = data.iloc[range(engRange[0][0], engRange[0][1])]
print(cfm56.head())

# Get ambient conditions 
[Tamb, Pamb] = ffms.FuelFlowMethods.ISA_conditions(alt = 0)
print(Tamb, Pamb)

# Get total conditions
speed = 0.4 # Mach
T1total = Tamb*(1+0.2*speed)
P1total = Pamb*(1+0.2*speed**2)**3.5

# Isentropic formula - Get isentropic temperature
gamma = 1.4
P2total = P1total*cfm56["Pressure Ratio"].values.astype(float)
T2is = T1total*(P2total/P1total)**((gamma-1)/gamma)
print(T2is)

# Get the real temperature
etac = 0.9 # Isentropic coefficient
T2total = T1total + (T2is-T1total)/etac
print(T2total)

# Add new feature to dataframe
cfm56["Combustor Inlet Temperature (K)"] = T2total
print(cfm56.head())


# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Define the operating points and their corresponding columns
ops = [
    ("Idle", "Fuel Flow Idle (kg/sec)", "NOx EI Idle (g/kg)"),
    ("T/O", "Fuel Flow T/O (kg/sec)", "NOx EI T/O (g/kg)"),
    ("C/O", "Fuel Flow C/O (kg/sec)", "NOx EI C/O (g/kg)"),
    ("App", "Fuel Flow App (kg/sec)", "NOx EI App (g/kg)")
]
colors = ['r', 'g', 'b', 'm']

for op, color in zip(ops, colors):
    label, fuel_col, ei_col = op
    ax.scatter(
        cfm56[fuel_col].astype(float),
        cfm56["Pressure Ratio"].astype(float),
        cfm56[ei_col].astype(float),
        c=color, label=label
    )

ax.set_xlabel("Fuel Flow (kg/sec)")
ax.set_ylabel("OPR (Pressure Ratio)")
ax.set_zlabel("NOx EI (g/kg)")
ax.set_title("NOx Emission Index vs Fuel Flow and OPR")
ax.legend()
plt.tight_layout()
#plt.show()


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Define the operating points and their corresponding columns
ops = [
    ("Idle", "Fuel Flow Idle (kg/sec)", "NOx EI Idle (g/kg)"),
    ("T/O", "Fuel Flow T/O (kg/sec)", "NOx EI T/O (g/kg)"),
    ("C/O", "Fuel Flow C/O (kg/sec)", "NOx EI C/O (g/kg)"),
    ("App", "Fuel Flow App (kg/sec)", "NOx EI App (g/kg)")
]
colors = ['r', 'g', 'b', 'm']

for op, color in zip(ops, colors):
    label, fuel_col, ei_col = op
    # Calculate Fuel Flow over Rated Thrust
    x = cfm56[fuel_col].astype(float) / cfm56["Rated Thrust (kN)"].astype(float)
    y = cfm56["Pressure Ratio"].astype(float)
    z = cfm56[ei_col].astype(float)
    ax.scatter(
        x,
        y,
        z,
        c=color, label=label
    )

ax.set_xlabel("Fuel Flow / Rated Thrust (kg/sec per kN)")
ax.set_ylabel("OPR (Pressure Ratio)")
ax.set_zlabel("NOx EI (g/kg)")
ax.set_title("NOx Emission Index vs (Fuel Flow / Rated Thrust) and OPR")
ax.legend()
plt.tight_layout()
plt.show()