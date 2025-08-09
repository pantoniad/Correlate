import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt    

####

def longFormatCV(dataRange, EIs, clmns):

    """
    
    """

    # 
    data = EIs.iloc[range(dataRange[0][0], dataRange[0][1])]
    data = data.reset_index()

    # 
    for i in clmns:

        # 
        pollutant_dots = data[i].astype(float)
        pollutant_dotsa = np.sort(pollutant_dots.values)

        if i == "CO EI T/O (g/kg)":
            pollutantTO = pollutant_dotsa
        elif i == "CO EI Idle (g/kg)":
            pollutantIdle = pollutant_dotsa
        elif i == "CO EI App (g/kg)":
            pollutantApp = pollutant_dotsa
        elif i == "CO EI C/O (g/kg)":
            pollutantCO = pollutant_dotsa
    
    offset = 0
    plots = []

    # Shift HC values
    Too = pollutantTO + offset
    plots.append(("T/O", Too))

    # Shift CO values
    Idle = pollutantIdle + offset
    plots.append(("Idle", Idle))
    offset = Idle.max() + 2

    # Shift NOx values
    App = pollutantApp + offset
    plots.append(("App", App))
    offset = App.max() + 2  # +2 for a gap
    
    # Shift NOx values
    Coo = pollutantCO + offset
    plots.append(("CO", Coo))

    df_all = pd.DataFrame({
        "Pollutant": np.concatenate([[p[0]] * len(p[1]) for p in plots]),
        "Value": np.concatenate([p[1] for p in plots])
    })

    return df_all

####
## Data insertion
data_og = pd.read_csv(r"E:/Correlate/Databank/ICAO_data.csv", delimiter=";")

clmns = ["HC EI Idle (g/kg)", "HC EI T/O (g/kg)", "HC EI C/O (g/kg)", "HC EI App (g/kg)"]
EIs = data_og[clmns]
cfm56_range = [[61, 169]]

# CF-56 and variants during Take-off
cfm56Data = EIs.iloc[range(cfm56_range[0][0],cfm56_range[0][1])]
cfm56Data = cfm56Data.reset_index()

cfm56COIdle_dots = cfm56Data[clmns[0]].astype(float) 
cfm56COIdle_dotsa = np.sort(cfm56COIdle_dots.values)
meanIdle = np.mean(cfm56COIdle_dotsa)

cfm56COTO_dots = cfm56Data[clmns[1]].astype(float) 
cfm56COTO_dotsa = np.sort(cfm56COTO_dots.values)
meanTO = np.mean(cfm56COTO_dotsa)

cfm56COCO_dots = cfm56Data[clmns[2]].astype(float) 
cfm56COCO_dotsa = np.sort(cfm56COCO_dots.values)
meanCO = np.mean(cfm56COCO_dotsa)

cfm56COApp_dots = cfm56Data[clmns[3]].astype(float) 
cfm56COApp_dotsa = np.sort(cfm56COApp_dots.values)
meanApp = np.mean(cfm56COApp_dotsa)

# # # 
plots = []
labels = ["HC Idle", "HC T/O", "HC C/O", "HC App"]

# Shift HC values
co_idle = cfm56COIdle_dotsa
plots.append((labels[0], co_idle))

# Shift CO values
co_to = cfm56COTO_dotsa
plots.append((labels[1], co_to))
offset = co_to.max() + 2

# Shift NOx values
co_co = cfm56COCO_dotsa
plots.append((labels[2], co_co))
offset = co_co.max() + 2  # +2 for a gap

# Shift NOx values
co_app = cfm56COApp_dotsa 
plots.append((labels[3], co_app))
offset = co_app.max() + 2  # +2 for a gap


# Combine into one DataFrame
df_all = pd.DataFrame({
    "Pollutant": np.concatenate([[p[0]] * len(p[1]) for p in plots]),
    "Value": np.concatenate([p[1] for p in plots])
})

# Custom colors for pollutants
fig1 = plt.figure(figsize=(7,5))
palette = {
    labels[0]: "royalblue",
    labels[1]: "green",
    labels[2]: "red",
    labels[3]: "magenta"
}

ax = sns.stripplot(
    data=df_all, 
    x="Pollutant", 
    y="Value", 
    jitter=0.1, 
    size=5, 
    palette=palette
)

mean_points = pd.DataFrame({
    "Names": [labels[0], labels[1], labels[2], labels[3]],
    "Values": [meanIdle, meanTO, meanCO, meanApp]
})

plt.plot(
    mean_points["Names"], 
    mean_points["Values"], 
    "--*", 
    color = "k", 
    zorder = 10,
    label = "Mean values"
)

plt.grid()
plt.legend()
plt.ylabel("Emissions Index Value (g/kg)")
plt.xlabel("Pollutant and operating point")
plt.title("HC EI over engine operation points - Dot plot - CFM56 family")
plt.yticks([0, 2, 4, 6, 8, 10])

fig2 = plt.figure(figsize=(7,5))
sns.swarmplot(
    data=df_all[::2],
    x="Pollutant",
    y="Value",
    size=5,
    palette=palette
)


plt.plot(
    mean_points["Names"], 
    mean_points["Values"], 
    "--*", 
    color = "k", 
    zorder = 10,
    label = "Mean values"
)

plt.grid()
plt.legend()
plt.ylabel("Emissions Index Value (g/kg)")
plt.xlabel("Pollutant and operating point")
plt.title("HC EI over engine operation points - Bee-swarm plot - CFM56 family")
plt.yticks([0, 2, 4, 6, 8, 10])


plt.show()
