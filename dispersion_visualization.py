import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt    

## Data insertion
data_og = pd.read_csv(r"E:/Computational_results/Databank_and_correlations_comparison/Databank/ICAO_data.csv", delimiter=";")

# Extract data - List of columns
clmns = ["Pressure Ratio", "B/P Ratio", "Rated Thrust (kN)", "Ambient Baro Min (kPa)",
         "CO EI T/O (g/kg)",  "CO EI C/O (g/kg)", "CO EI App (g/kg)", "CO EI Idle (g/kg)",
         "HC EI T/O (g/kg)", "HC EI C/O (g/kg)", "HC EI App (g/kg)", "HC EI Idle (g/kg)",
         "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)",  "Fuel Flow App (kg/sec)", "Fuel Flow Idle (kg/sec)",
         "NOx EI T/O (g/kg)",  "NOx EI C/O (g/kg)",  "NOx EI App (g/kg)",  "NOx EI Idle (g/kg)", "NOx LTO Total mass (g)"]

# CF-56 and variants
cf56_range = [[61, 169]]
cfm56Data = data_og.iloc[cf56_range[0][0]:cf56_range[0][1], 42:43]
freq_table = cfm56Data.value_counts()


cfm56NOx_dots = cfm56Data["NOx EI T/O (g/kg)"].astype(float) 
cfm56NOx_dotsa = np.sort(cfm56NOx_dots.values)
cfm56NOx_labels = np.full(len(cfm56NOx_dotsa),"NOx T/O")
cfm56NOx = np.vstack((cfm56NOx_labels, cfm56NOx_dotsa))
print(cfm56NOx)

sns.stripplot(x = cfm56NOx[0,:], y = cfm56NOx[1,:], jitter = 0.1, size = 5)  
plt.grid()
plt.show()