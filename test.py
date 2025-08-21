# Swarm plot
fig2 = plt.figure(figsize=(9,7))
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
    markersize = 10,
    color = "k", 
    zorder = 10,
    label = "Mean values - ICAO Databank"
)

plt.plot(
    labels,
    dtCorrs.iloc[:]["Rokket"],
    "-->",
    #color = "orangered",
    label = "Rokke"
)

plt.plot(
    labels,
    dtCorrs.iloc[:]["Novelo"],
    ":1",
    #color = "violet",
    label = "Novelo",
    zorder = 10
)

plt.plot(
    labels,
    dtCorrs.iloc[:]["Kyprianidis"],
    ":<",
    #color = "indigo",
    label = "Kyprianidis",
    zorder = 10
)

plt.plot(
    labels,
    dtCorrs.iloc[:]["Lewis"],
    ":+",
    #color = "magenta",
    label = "Lewis",
    zorder = 10
)

plt.plot(
    labels,
    dtCorrs.iloc[:]["Perkavec"],
    "--o",
    #color = "purple",
    label = "Perkavec"
)

plt.plot(
    labels, 
    exp["Turgut - CFM56-7B26"],
    "-8",
    label = "Turgut, CFM56-7B26",
    zorder = 10
)

plt.grid(color = "silver", linestyle = ":")
plt.legend()
plt.ylabel("Emissions Index Value (g/kg)")
plt.xlabel("Pollutant and operating point")
plt.title("NOx EI over engine operation points - Bee-swarm plot - CFM56 family")
plt.yticks(range(0,90,10))


plt.show()
