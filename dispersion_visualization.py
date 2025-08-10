import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt    
import correlations_class as correlate

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

clmns = ["NOx EI Idle (g/kg)", "NOx EI T/O (g/kg)", "NOx EI C/O (g/kg)", "NOx EI App (g/kg)"]
EIs = data_og[clmns]
cfm56_range = [[61, 169]]

# CF-56 and variants
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

# Getting data ready for plotting
plots = []
labels = ["NOx Idle", "NOx T/O", "NOx C/O", "NOx App"]

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

### Correlation equations ###
# Dictionary definition: For every key: Tburner_inlet Tburner_outlet, Pburner, m_dot, FAR
d = {
    "idle": [600, 1200, 3156.71, 23.02, 0.0139],
    "take-off": [860, 2250, 3152.59, 54.1, 0.0214],
    "climb-out": [820, 2100, 3240.46, 52.41, 0.0306],
    "approach": [750, 1400, 2909.36, 36.79, 0.01288]
}

dtPoints = pd.DataFrame(
    data = d, 
    index = ["Tbin", "Tbout", "Pbin", "m_dot", "FAR"]
)

dtCorrs = pd.DataFrame([])

# Correlations instance
for point in dtPoints.keys():

    # Create class instance for opeating point
    corr = correlate.Correlations(
        dtPoints[point]["Tbin"], 
        dtPoints[point]["Tbout"], 
        dtPoints[point]["Pbin"], 
        0.95*dtPoints[point]["Pbin"], 
        dtPoints[point]["FAR"], 
        dtPoints[point]["m_dot"], 
        0, 
        1.293
    )

    # Get values from correlation equations
    becker = corr.becker(1600, method = "simplified")
    rokke = corr.rokke_nox(41, method = "Index")
    lewis = corr.lewis_nox()
    kyprianidis = corr.kyprianidis(h = 0)
    novelo = corr.novelo()
    perkavec = corr.perkavec()

    # Create temporary dataframe
    d = {
        "Becker": becker,
        "Rokket": rokke,
        "Lewis": lewis,
        "Kyprianidis": kyprianidis,
        "Novelo": novelo,
        "Perkavec": perkavec
    }

    index = [point]

    dt1 = pd.DataFrame(
        data = d,
        index = index
    )

    # Append temporary dataframe to external
    dtCorrs = pd.concat([dtCorrs, dt1], axis = 0)

print(dtCorrs)


# Plotting

# Dot plot
fig1 = plt.figure(figsize=(7,5))
palette = {
    labels[0]: "royalblue",
    labels[1]: "green",
    labels[2]: "red",
    labels[3]: "magenta"
}

# Create the dot plot
ax = sns.stripplot(
    data=df_all, 
    x="Pollutant", 
    y="Value", 
    jitter=0.1, 
    size=5, 
    palette=palette
)

# Include the mean values
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

# Include values from Correlaion equations
#plt.plot(
#    labels,
#    dtCorrs.iloc[:]["Becker"],
#    "--s",
#    color = "orangered",
#    label = "Becker"
#)

#plt.plot(
#    labels,
#    dtCorrs.iloc[:]["Rokket"],
#    "-->",
#    #color = "orangered",
#    label = "Rokket"
#)

plt.plot(
    labels,
    dtCorrs.iloc[:]["Novelo"],
    ":1",
    #color = "violet",
    label = "Novelo"
)

plt.plot(
    labels,
    dtCorrs.iloc[:]["Kyprianidis"],
    ":<",
    #color = "indigo",
    label = "Kyprianidis"
)

plt.plot(
    labels,
    dtCorrs.iloc[:]["Lewis"],
    ":+",
    #color = "magenta",
    label = "Lewis"
)

plt.plot(
    labels,
    dtCorrs.iloc[:]["Perkavec"],
    "--o",
    #color = "purple",
    label = "Perkavec"
)

plt.grid(color = "silver", linestyle = ":")
plt.legend()
plt.ylabel("Emissions Index Value (g/kg)")
plt.xlabel("Pollutant and operating point")
plt.title("NOx EI over engine operation points - Dot plot - CFM56 family")
plt.yticks(range(0,50,10))


# Swarm plot
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

plt.plot(
    labels,
    dtCorrs.iloc[:]["Novelo"],
    ":1",
    #color = "violet",
    label = "Novelo"
)

plt.plot(
    labels,
    dtCorrs.iloc[:]["Kyprianidis"],
    ":<",
    #color = "indigo",
    label = "Kyprianidis"
)

plt.plot(
    labels,
    dtCorrs.iloc[:]["Lewis"],
    ":+",
    #color = "magenta",
    label = "Lewis"
)

plt.plot(
    labels,
    dtCorrs.iloc[:]["Perkavec"],
    "--o",
    #color = "purple",
    label = "Perkavec"
)

plt.grid(color = "silver", linestyle = ":")
plt.legend()
plt.ylabel("Emissions Index Value (g/kg)")
plt.xlabel("Pollutant and operating point")
plt.title("NOx EI over engine operation points - Bee-swarm plot - CFM56 family")
plt.yticks(range(0,50,10))


plt.show()



