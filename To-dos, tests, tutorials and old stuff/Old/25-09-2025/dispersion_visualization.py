import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt  
import matplotlib.gridspec as gridspec  
import warnings
from Classes import correlations_class as correlate
from Classes.latex_class import latex as lx
from Classes.FuelFlow_class import FuelFlowMethods as ffms

####
def distribution_plots(df_all, mean_points, dtCorrs, exp, meanEC, meanEE, relativeEC, relativeEE, method, size, title, ylimits, xLabel, yLabel, colours, labels, dotPlotXlabel, dotPlotYlabel, lineStyle, Jitter = None):
    """
    distribution_plots: a function that is able to generate four kinds of plots based on the 
                        string given in the "method" parameter of the function. 
                        Supported DISTRIBUTION plots: Dot plots, Box plots, Violin Plots, Swarm plots

    Inputs:
    - df_all:   data to be placed as dots in dot plot
                Dataframe, first column are the names
                for each data position, second column 
                are the values
    - mean_points:  mean values from the ICAO Datapoints,
                    Dataframe, First column are the names 
                    for each data position of the dot plot,
                    Second column are the values for each
                    data position
    - dtCorrs:  Data retrieved from the usage of 
                correlation equations. Dataframe,
                X axis contains the names of the 
                authors of the correlations, Y 
                axis contains the EI values for 
                the four ICAO LTO points,
    - exp:  experimental data, Dataframe, X axis contains
            the source of the data, Y axis contains the EI
            values for the four LTO cycle points. For now
            able to integrate only one point,
    - meanEC: mean error and standard deviation values for the data 
              retrieved from correlation equations. Dataframe,
    - meanEE: mean error and standard deviation values from the data
              retrieved from experimental data. Dataframe,
    - relativeEC: relative error between the values of correlation equations
                  and the mean values of the ICAO data-points for each 
                  operating point, Dataframe
    - relativeEE: relative error between the values of experimental data
                  and the mean values of the ICAO data-points for each 
                  operating point, Dataframe
    - method:   used to determine the kind of distribution plot used
                Inputs: "Boxplot", "Dotplot", "Swarmplot", "Violinplot"
    - size: figure size. List,
    - title: title of the plot,
    - ylimits:  the limits of the y axis of the dot plot,
                List, [min, max, step]
    - xLabel: name of the x axis of the (whole) plot,
    - yLabel: name of the y axis of the (whole) plot
    - colours: colour palette for the dot plot. List
    - labels: Dot plot labels list,
    - Jitter: define the jitter,
    - dotPlotXlabel: the label for the x axis 
                     of the dot plot,
    - dotPlotYlabel: the label of the y axis
                     of the dot plot
    - lineStyle:  marker type for the plots of the
                    data calculated from the usage of 
                    correlation equations, list 

    Outputs:
    -   Based on the string provided for the method parameter
        a Dot, Swarm, Violin or Box plot is generated
    
    """
    # Valid distribution plot methods 
    valid_methods = ["Boxplot", "Swarmplot", "Dotplot", "Violinplot"]

    # Create palette dictionary
    paletteDict = {
        labels[0]: colours[0],
        labels[1]: colours[1],
        labels[2]: colours[2],
        labels[3]: colours[3]
    }

    # Create figure
    fig = plt.figure(figsize = (size[0], size[1]))
    #gs = gridspec.GridSpec(3, 1, height_ratios=[9, 1, 1], figure=fig)

    # Create dot plot
    if method == "Dotplot":

        if Jitter is None:
            
            # Default jitter value
            Jitter = 0.1

            warnings.warn(
                f"The 'Jitter' parameter must be provided. Assigning default value: {Jitter}.",
                UserWarning
                )
        
        #ax1 = fig.add_subplot(gs[0])
        ax1 = sns.stripplot(
            data = df_all,
            size = 5,
            x = dotPlotXlabel,
            hue = dotPlotXlabel,
            y = dotPlotYlabel,
            legend = False,
            palette = paletteDict
        )

        #ax.set(xticklabels = [])
        #ax.set(xlabel = None)

    # Create swarm plot
    elif method == "Swarmplot":
        ax = sns.swarmplot(
            data = df_all,
            x = dotPlotXlabel,
            y = dotPlotYlabel,
            palette = paletteDict
        )

    # Create boxplot
    elif method == "Boxplot":
        ax = sns.boxplot(
            data = df_all,
            x = dotPlotXlabel,
            y = dotPlotYlabel,
            palette = paletteDict
        )

    # Create violin plot
    elif method == "Violinplot":
        ax = sns.violinplot(
            data = df_all,
            x = dotPlotXlabel,
            y = dotPlotYlabel,
            palette = paletteDict
        )

    else:
        if not method:
            raise ValueError("No method provided. Please specify one of: "
                            + ", ".join(valid_methods))
        else:
            raise ValueError(f"Invalid method '{method}'. Choose from: "
                            + ", ".join(valid_methods))
       
    # Mean value plotting
    plt.plot(
        mean_points.index,
        mean_points.values,      
        "--*",
        markersize = 10,
        color = "black",
        zorder = 10,
        label = "Mean values - ICAO Databank"
    )

    
    # Correlation equations value plotting
    pointer = 0
    for i in dtCorrs.keys():
        
        # Add the data to the plot
        plt.plot(
            labels, 
            dtCorrs.iloc[:][i],
            lineStyle[pointer], 
            label = i 
        )

        # Increase the count of the pointer
        pointer += pointer
    
    # Place the experimental data
    plt.plot(
        labels, 
        exp["Turgut - CFM56-7B26"],
        "-8",
        label = "Turgut, CFM56-7B26",
        zorder = 10
    )

    # Additional plot settings, Show plot
    plt.grid(color = "silver", linestyle = ":")
    plt.legend(loc = "best")
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.title(title)
    plt.yticks(range(ylimits[0], ylimits[1], ylimits[2]))

    plt.show()

def error(dtCorrs, mean_points, exp):
    """
    error:  function to calculate the mean relative error between
            the correlation equations and experimental results
            with the mean points of the data for all of the operating
            points. Calculates the relative error between each
            correlation and experimental data point with the mean 
            for each operating point. Then regroups the data and 
            calculates the mean value across all operating points. 

            Can be used with any number of correlation equations and 
            experimental data sets. 

    Inputs:
    - mean_points: the mean values from the ICAO data set. Dataframe,
    - dtCorrs: the values of the correlation equations for each operating
                point. Dataframe
    - exp: the experimental data retrieved from literature. Dataframe

    Outputs:
    - meanRelativeEC: the mean relative error between all of the correlation
                        equations and the mean values of each operating point, 
                        for all operating points, Dataframe 
    - meanRelativeEE: same with the above, but for the experimental datasets, 
                        Dataframe
    """
    
    # Constants and parameters
    operPoints = len(mean_points.index)
    corrsNum = len(dtCorrs.keys())
    expNum = len(exp.keys())
    relativeEC = [] # relative error for Correlation
    relativeEE = [] # relative error for Experimental 
    meanRelativeEC = [] # mean relative error for each correlation
    meanRelativeEE = [] # mean relative error for Experimental

    # Calculate relative error
    for i in range(0, operPoints):

        # Get parameters 
        mean = mean_points.values[i]

        # Calculate relative error - Correlation equations
        for j in range(0, corrsNum):
            
            # Calculate the relative error for each correlation j
            # at each operating point i
            error = 100*np.abs(dtCorrs.values[i][j] - mean)/mean

            # Append the value to a bigger dictionary 
            relativeEC = np.append(relativeEC, error, axis = 0)

        for j in range(0, expNum):
        
            # Calculate the relative error for each correlation j
            # at each operating point i
            error = 100*np.abs(exp.values[i][j] - mean)/mean

            # Append the value to a bigger dictionary 
            relativeEE = np.append(relativeEE, error, axis = 0)

    # Regroup the relative error lists to be 2d arrays
    chunks = [relativeEC[i:i+corrsNum] for i in range(0, len(relativeEC), corrsNum)]
    relativeECr = list(map(list, zip(*chunks)))
    
    chunks = [relativeEE[i:i+corrsNum] for i in range(0, len(relativeEE), expNum)]
    relativeEEr = list(map(list, zip(*chunks)))

    # Get mean values
    meanRelativeEC = [np.mean(relativeECr[:][i]) for i in range(0, corrsNum)]
    meanRelativeEE = [np.mean(relativeEEr[:][i]) for i in range(0, expNum)]

    # Convert into data-frames
    meanEC = pd.DataFrame(
        data = np.round(meanRelativeEC,2),
        columns = ["Mean Relative percentage error"],
        index = dtCorrs.keys()
    )

    meanEE = pd.DataFrame(
        data = np.round(meanRelativeEE,2),
        columns = ["Mean relative percentage error"],
        index = exp.keys()
    )

    # Also convert the relative error lists
    lis = np.array(relativeECr)
    lisT = lis.T
    listT = lisT.tolist()
    relativeECd = pd.DataFrame(
        data = np.round(listT,2),
        columns = dtCorrs.keys(),
        index = dtCorrs.index
    )
    
    lis = np.array(relativeEEr)
    lisT = lis.T
    listT = lisT.tolist()
    relativeEEd = pd.DataFrame(
        data = np.round(listT,2),
        columns = exp.keys(),
        index = exp.index
    )
     
    # Get standard deviations
    stdEC = np.round([np.std(relativeECr[:][i]) for i in range(0, corrsNum)],2)
    stdEE = np.round([np.std(relativeEEr[:][i]) for i in range(0, expNum)],2)

    # Include the standard deviation 
    meanEE["Standard Deviation"] = stdEE
    meanEC["Standard Deviation"] = stdEC
    
    return meanEC, meanEE, relativeECd, relativeEEd

### Scripting ###
## Data insertion
data_og = pd.read_csv(r"E:/Correlate/Databank/ICAO_data.csv", delimiter=";")

clmns = ["NOx EI Idle (g/kg)", "NOx EI T/O (g/kg)", "NOx EI C/O (g/kg)", "NOx EI App (g/kg)"]
EIs = data_og[clmns]
cfm56_range = [[61, 169]]

# CF-56 and variants
cfm56Data = EIs.iloc[range(cfm56_range[0][0],cfm56_range[0][1])]
cfm56Data = cfm56Data.reset_index()

cfm56Idle_dots = cfm56Data[clmns[0]].astype(float) 
cfm56Idle_dotsa = np.sort(cfm56Idle_dots.values)
meanIdle = np.mean(cfm56Idle_dotsa)

cfm56TO_dots = cfm56Data[clmns[1]].astype(float) 
cfm56TO_dotsa = np.sort(cfm56TO_dots.values)
meanTO = np.mean(cfm56TO_dotsa)

cfm56CO_dots = cfm56Data[clmns[2]].astype(float) 
cfm56CO_dotsa = np.sort(cfm56CO_dots.values)
meanCO = np.mean(cfm56CO_dotsa)

cfm56App_dots = cfm56Data[clmns[3]].astype(float) 
cfm56App_dotsa = np.sort(cfm56App_dots.values)
meanApp = np.mean(cfm56App_dotsa)

# Getting data ready for plotting
plots = []
labels = ["NOx Idle", "NOx T/O", "NOx C/O", "NOx App"]

# Shift HC values
idle = cfm56Idle_dotsa
plots.append((labels[0], idle))

# Shift CO values
to = cfm56TO_dotsa
plots.append((labels[1], to))
offset = to.max() + 2

# Shift NOx values
co = cfm56CO_dotsa
plots.append((labels[2], co))
offset = co.max() + 2  # +2 for a gap

# Shift NOx values
app = cfm56App_dotsa 
plots.append((labels[3], app))
offset = app.max() + 2  # +2 for a gap


# Combine into one DataFrame
df_all = pd.DataFrame({
    "Pollutant": np.concatenate([[p[0]] * len(p[1]) for p in plots]),
    "Value": np.concatenate([p[1] for p in plots])
})

### Correlation equations ###
# Dictionary definition: For every key: Burner Inlet temperature (K), 
# Burner outlet temperature (K),
# Burner inlet pressure (Pa),
# Core air mass flow rate (kg/s), 
# FAR
d = {
    "idle": [797.1, 1290, 2755850, 10.564, 0.0137],
    "take-off": [809.95, 2250, 2929690, 46.897, 0.0446],
    "climb-out": [805.1, 2000, 2828980, 45.44, 0.03596],
    "approach": [787.91, 1400, 2539920, 31.89, 0.01718]
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
eisff = data_og[clmns2]
cfm56_7b26 = [[135, 136]]

engineData = eisff.iloc[range(cfm56_7b26[0][0], cfm56_7b26[0][1])]

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
#print(einox)

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

print(exp)

# Dot plot #
# Colour palette
palette = ["royalblue", "green", "red", "magenta"]

# Marker styles
lineStyle = ["-->", ":1", ":<", ":+", "-8"]

# Include the mean values
d = {
    "NOx": [meanIdle, meanTO, meanCO, meanApp],
}

mean_points = pd.DataFrame(
    data = d,
    index = labels
)

mean_lx = lx(df = mean_points, filename = "data/means.tex", caption = "Operating points - Mean values", label = "tab:means")
mean_lx.df_to_lxTable()

print(mean_points)

# Mean relative error and standard deviation 
[meanEC, meanEE, relativeEC, relativeEE] = error(mean_points=mean_points, dtCorrs=dtCorrs, exp = exp)

print(meanEC)
print(meanEE)
print(relativeEC)
print(relativeEE)

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

# Engine specifications
d = {
    "Thrust rating (kN)": [117],
    "Fan diameter": [1.55],
    "Hub2Tip": [0.3],
    "Bypass ratio": [5.1],
    "Fan PR": [1.6],
    "Booster PR": [1.55],
    "High pressure compressor PR": [11]
}

specs = pd.DataFrame(
    data = d,
    index = ["Parameter", "Value"]
)

engineSpecs = lx(df = specs.T, filename = "data/specs.tex", caption = "CFM56-7B26 specifications",
                 label = "tab:specs")
engineSpecs.df_to_lxTable()

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
distribution_plots(
    df_all, 
    mean_points, 
    dtCorrs,
    exp,
    meanEC,
    meanEE,
    relativeEC,
    relativeEE,
    method = "Swarmplot",
    size = [12,9],
    ylimits = [0, 70, 10], # min, max, step 
    title = "NOx EI over engine operation points - Dot plot - CFM56 family", 
    xLabel = "Pollutant and operating point", 
    yLabel = "Emissions index value (g/kg)", 
    colours = palette, 
    labels = labels,  
    dotPlotXlabel = "Pollutant", 
    dotPlotYlabel = "Value", 
    lineStyle = lineStyle
)