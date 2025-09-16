import pandas as pd
import numpy as np

from Classes.data_processor_class import data_process
from Classes.models_class import models_per_OP
from Classes.data_plotting_class import data_plotting
import Classes.correlations_class as correlate
from Classes.FuelFlow_class import FuelFlowMethods as ffms
from Classes.latex_class import latex as lx

## Data used on the script ##

# Load ICAO data
df = pd.read_csv(r"Databank/ICAO_data.csv", delimiter = ";")

clmns = ["Pressure Ratio", "Rated Thrust (kN)", "Fuel Flow Idle (kg/sec)", 
         "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)","Fuel Flow App (kg/sec)",
         "NOx EI Idle (g/kg)", "NOx EI T/O (g/kg)", "NOx EI C/O (g/kg)",
         "NOx EI App (g/kg)"]

drange = [[61, 169]]

dfCleanUp = data_process(df = df, clmns = clmns, drange = drange)
df = dfCleanUp.csv_cleanup(reset_index = True, save_to_csv = True, path = "Databank/CFM56data.csv")

ops = ["Idle", "T/O", "C/O", "App"]

# Results from thermodynamic model
# Col1: Tin, Col2: Tout, Co3: Pin (Pa), Col4: m_dot_core, Col5: m_dot_fuel (kg/s)
d = {
    ops[0]: [797.1, 1290, 2755850, 10.564, 0.0137, 0.144],
    ops[1]: [809.95, 2250, 2929690, 46.897, 0.0446, 2.02],
    ops[2]: [805.1, 2000, 2828980, 45.44, 0.03596, 1.634],
    ops[3]: [787.91, 1400, 2539920, 31.89, 0.01718, 0.548]
}

dtPoints = pd.DataFrame(
    data = d, 
    index = ["Tbin", "Tbout", "Pbin", "m_dot_air", "FAR", "m_dot_fuel"]
)

# Reference engine data - CFM56-7B26
# ICAO fuel flow and EINOx
cfm56_7b26 = [[74, 75]]

engineData = df.iloc[range(cfm56_7b26[0][0], cfm56_7b26[0][1])]
engineData = df.drop(["Pressure Ratio", "Rated Thrust (kN)"], axis = 1)

# Engine specifications
d = {
    "Thrust rating (kN)": [117],
    "Fan diameter": [1.55],
    "Hub2Tip": [0.3],
    "Bypass ratio": [5.1],
    "Fan PR": [1.6],
    "Booster PR": [1.55],
    "High pressure compressor PR": [11],
    "Pressure ratio": [27.6]
}

specs = pd.DataFrame(
    data = d,
    index = ["Value"]
)


#engineSpecs = lx(df = specs.T, filename = "data/specs.tex", caption = "CFM56-7B26 specifications",
#                 label = "tab:specs")
#engineSpecs.df_to_lxTable()


## Model execution and generation ##

# Saving the model results
models_res = {
    "Linear Regression": {"Idle": [], "T/O": [], "C/O": [], "App": []},
    "Gradient Boosting": {"Idle": [], "T/O": [], "C/O": [], "App": []},
    "ANN": {"Idle": [], "T/O": [], "C/O": [], "App": []}
} 

# Iterate through the opeating points
for i in ops:
    
    print(i)
    # Keep only columns that contain the operating point 
    df1 = df.filter(df.columns[df.columns.str.contains(i)], axis=1)
    
    # Get the other columns and append
    df2 = df.filter(["Pressure Ratio", "Rated Thrust (kN)"])
    df3 = pd.concat([df2, df1], axis = 1)

    # Get features and response
    features = df3.drop(columns=f"NOx EI {i} (g/kg)")
    
    if i == ops[0]:
        features["Rated Thrust (kN)"] = 0.07*features["Rated Thrust (kN)"].values.astype(float)
        x_new = [specs["Pressure ratio"]["Value"].astype(float), 0.07*specs["Thrust rating (kN)"]["Value"].astype(float), dtPoints[i]["m_dot_fuel"]]
    elif i == ops[1]:
        pass
    elif i == ops[2]:
        features["Rated Thrust (kN)"] = 0.85*features["Rated Thrust (kN)"].values.astype(float)
        x_new = [specs["Pressure ratio"]["Value"].astype(float), 0.85*specs["Thrust rating (kN)"]["Value"].astype(float), dtPoints[i]["m_dot_fuel"]]
    elif i == ops[3]:
        features["Rated Thrust (kN)"] = 0.3*features["Rated Thrust (kN)"].values.astype(float)
        x_new = [specs["Pressure ratio"]["Value"].astype(float), 0.3*specs["Thrust rating (kN)"]["Value"].astype(float), dtPoints[i]["m_dot_fuel"]]

    response = df3[f"NOx EI {i} (g/kg)"]

    # Initialize models_per_OP class
    models = models_per_OP(
        data = df3,
        features = features,
        response = response
    )

    # Split data
    X_train, y_train, X_dev, y_dev, X_test, y_test = models.splitter(
        train_split = 0.51,
        test_split = 0.15,
        dev_split = 0.34
    )

    # Train on the dev set (only applicable to Polynomial regression as of now)
    parameters = {"Degrees": 2, "Include Bias": False}
    polymodel, polyfeatures, train_poly, test_poly = models.polReg(
        xtrain = X_train, ytrain = y_train, xtest = X_dev, ytest = y_dev,
        parameters = parameters
    )
    
    # Get metrics
    metrics = models.performance_metrics(train = train_poly, test = test_poly)
    print(f"Operating point: {i} metrics")
    print(metrics.head())

    # Predict based on the thermodynamic data
    #x_new = [specs["Pressure ratio"]["Value"].astype(float), 0.07*specs["Thrust rating (kN)"]["Value"].astype(float), dtPoints[i]["m_dot_fuel"]]
    x_new_poly = polyfeatures.transform([x_new])
    y_new = polymodel.predict(x_new_poly)
    
    # Save prediction results
    models_res["Linear Regression"][i] = y_new

    # Learning curve
    models.Learning_curve(model = polymodel, model_features = polyfeatures, 
                          operating_point = i)

# Convert models_res to dataframe
filtered = {
    model: {phase: vals for phase, vals in phases.items() if vals} 
    for model, phases in models_res.items()
    if any(phases.values())  # keep only models with at least one non-empty entry
}

dtmodels = pd.DataFrame(filtered)

### Distribution plots
labels = ["NOx Idle", "NOx T/O", "NOx C/O", "NOx App"]

## Create the df_all for the distribution_plots method
dfa = df.filter(df.columns[df.columns.str.contains("NOx")], axis = 1)

plots = []
for i in ops:
    dfb = dfa.filter(df.columns[df.columns.str.contains(i)], axis = 1)

    dfb_values = dfb[f"NOx EI {i} (g/kg)"].astype(float).values
    plots.append((f"NOx {i}", np.sort(dfb_values)))

df_all = pd.DataFrame({
    "Pollutant": np.concatenate([[p[0]] * len(p[1]) for p in plots]),
    "Value": np.concatenate([p[1] for p in plots])
})

### Correlation equations ###
# Empty correlations dataframe
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
        dtPoints[point]["m_dot_air"], 
        0, 
        1.293
    )

    # Get values from correlation equations
    becker = corr.becker(1800, method = "simplified")
    rokke = corr.rokke_nox(27.1, method = "Index")
    lewis = corr.lewis_nox()
    kyprianidis = corr.kyprianidis(h = 0)
    novelo = corr.novelo()
    perkavec = corr.perkavec()
    lefebvre = corr.lefebvre(Vc = 0.05, Tpz = 2000, Tst = 2250)
    gasturb = corr.gasturb(WAR = 0)
    ge = corr.generalElectric(WAR = 0)
    aeronox = corr.aeronox(Vc = 0.1, R = 287)

    # Create temporary dataframe
    d = {
        #"Becker": becker,
        #"Perkavec": perkavec,
        "Rokke": rokke,
        "Lewis": lewis,
        "Kyprianidis": kyprianidis,
        "Novelo": novelo,
        "Lefebvre": lefebvre,
        "GasTurb": gasturb,
        "General Electric": ge,
        "Aeronox": aeronox
    }

    index = [point]

    dt1 = pd.DataFrame(
        data = d,
        index = index
    )

    # Append temporary dataframe to external
    dtCorrs = pd.concat([dtCorrs, dt1], axis = 0)

# Fuel flow methods - DLR
clmns2 = ["NOx EI Idle (g/kg)", "NOx EI T/O (g/kg)", "NOx EI C/O (g/kg)", "NOx EI App (g/kg)", 
         "Fuel Flow Idle (kg/sec)", "Fuel Flow T/O (kg/sec)", "Fuel Flow C/O (kg/sec)", "Fuel Flow App (kg/sec)"]

# EIs and Fuel flow data from ICAO for all engines
eisff = df[clmns2]

# Operating conditions 
speed = [0, 0.4, 0.4, 0.3]  # Mach number
alt = [0, 11, 304, 905]     # meters

d = {
    "EINOx": engineData.iloc[0][0:4].values.astype(float),
    "Fuel Flows": engineData.iloc[0][4:8].values.astype(float),
    "Flight altitude": alt,
    "Flight Speed": speed
}

datapoints = pd.DataFrame(
    data = d,
    index = ["Idle", "Take-off", "Climb-out", "Approach"]
)
datapoints = datapoints.T

ff = ffms(datapoints = datapoints, fitting = "Parabolic", check_fit = False)
ffeinox = ff.dlrFF()

# Add fuel flow EIs to dtCorrs
dtCorrs["DLR Fuel Flow"] = ffeinox.values.T

# Experimental data insertion - Dataframe format
exp_data = {
    "Turgut - CFM56-7B26": [1.8, 24.4, (12.6+16.4)/2, 2.8],
    "Becker - PG6541B": [7.73*10**(-4)*30, 7.73*10**(-4)*80, 7.73*10**(-4)*260, 7.73*10**(-4)*300]
}

exp = pd.DataFrame(
    data = exp_data,
    index = dtPoints.keys()    
)

experimental = lx(df = exp, filename = "data/experimental.tex", caption = "Turgut et. al - CFM56-7B26", label = "tab:exp")
experimental.df_to_lxTable()

# Dot plot #
# Colour palette
palette = ["royalblue", "green", "red", "magenta"]

# Marker styles
lineStyle = ["-->", ":1", ":<", ":+", "-8"]

# Include the mean values
meanIdle = np.mean(df["NOx EI Idle (g/kg)"].values.astype(float))
meanTO = np.mean(df["NOx EI T/O (g/kg)"].values.astype(float))
meanCO = np.mean(df["NOx EI C/O (g/kg)"].values.astype(float))
meanApp = np.mean(df["NOx EI App (g/kg)"].values.astype(float))
d = {
    "NOx": [meanIdle, meanTO, meanCO, meanApp],
}

mean_points = pd.DataFrame(
    data = d,
    index = labels
)

mean_lx = lx(df = mean_points, filename = "data/means.tex", caption = "Operating points - Mean values", label = "tab:means")
mean_lx.df_to_lxTable()

# Mean relative error and standard deviation 
errors = data_plotting(df_all = df_all, dtCorrs = dtCorrs, exp = exp, mean_points = mean_points, dtmodels = dtmodels)
[meanEC, meanEE, relativeEC, relativeEE] = errors.error()

# Convert dataframes to latex tables
# Relative error - EC: correlation equations error, EE: experimental error
relativeEC = lx(df = relativeEC.T, filename = "data/relECerror.tex", caption = "Relative error between correalation results and ICAO mean value", label = "tab:relec")
relativeEC.df_to_lxTable()

relativeEE = lx(df = relativeEE.T, filename = "data/relEEerror.tex", caption = "Relative error between experimental data and ICAO men value", label = "tab:relee")
relativeEE.df_to_lxTable()

# Mean relative error - EC: Correlation equations error, EE: experimental error
meanEC = lx(df = meanEC, filename = "data/MEANEC.tex", caption = "Mean relative error - Correlation equations", label = "meanEC")
meanEC.df_to_lxTable()

meanEE = lx(df = meanEE, filename = "data/MEANEE.tex", caption = "Mean realtive error - Experimental data", label = "meanEE")
meanEE.df_to_lxTable()

# Values of thermodynamic parameters
dtPoints = lx(df = dtPoints, filename = "data/ops.tex", caption = "Values of thermodynamic parameters - LTO Cycle points", label = "tab:Thermo")
dtPoints.df_to_lxTable()

# Operating conditions for each point: Altitude, Required thrust, Flight speed, Axial fan speed
d = {
    "Idle": [0, 8.19, 0, 0.09],
    "Take-off": [11, 117, 0.3, 0.4],
    "Climb-out": [305, 99.45, 0.3, 0.4],
    "Approach": [914, 35.1, 0.2, 0.3]
}

lto_ops = pd.DataFrame(
    data = d,
    index = ["Altitude (m)", "Required thrust (kN)", "Flight speed (Mach)", "Axial fan speed (Mach)"]
)

conditions = lx(df = lto_ops, filename = "data/lto_ops.tex", 
                caption = "LTO operating conditions", label = "tab:lto")
conditions.df_to_lxTable()

# Distribution plots
distr_plots = data_plotting(df_all = df_all, dtCorrs = dtCorrs, exp = exp, mean_points = mean_points, dtmodels = dtmodels)
distr_plots.distribution_plots(
    method = "Violinplot",
    size = [12,9],
    #ylimits = [0, 70, 10], # min, max, step 
    title = "NOx EI over engine operation points - Dot plot - CFM56 family", 
    xLabel = "Pollutant and operating point", 
    yLabel = "Emissions index value (g/kg)", 
    colours = palette, 
    labels = labels,  
    dotPlotXlabel = "Pollutant", 
    dotPlotYlabel = "Value", 
    lineStyle = lineStyle
)
