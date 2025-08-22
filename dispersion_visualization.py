import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt    
import correlations_class as correlate
import warnings

####
def distribution_plots(df_all, mean_points, dtCorrs, exp, method, size, title, ylimits, xLabel, yLabel, colours, labels, dotPlotXlabel, dotPlotYlabel, lineStyle, Jitter = None):
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
            able to integrate only one point
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

    # Create dot plot
    if method == "Dotplot":

        if Jitter is None:
            
            # Default jitter value
            Jitter = 0.1

            warnings.warn(
                f"The 'Jitter' parameter must be provided. Assigning default value: {Jitter}.",
                UserWarning
                )
        
        ax = sns.stripplot(
            data = df_all,
            size = 5,
            x = dotPlotXlabel,
            hue = dotPlotXlabel,
            y = dotPlotYlabel,
            legend = False,
            palette = paletteDict
        )
    
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
        mean_points["Names"],
        mean_points["Values"],
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
        #"Becker": Becker,
        #"Perkavec": Perkavec,
        #"Rokke": rokke,
        "Lewis": lewis,
        "Kyprianidis": kyprianidis,
        "Novelo": novelo
    }

    index = [point]

    dt1 = pd.DataFrame(
        data = d,
        index = index
    )

    # Append temporary dataframe to external
    dtCorrs = pd.concat([dtCorrs, dt1], axis = 0)

# Experimental data insertion - Dataframe format
exp_data = {
    "Turgut - CFM56-7B26": [1.8, 24.4, (12.6+16.4)/2, 2.8],
    "Becker - PG6541B": [7.73*10**(-4)*30, 7.73*10**(-4)*80, 7.73*10**(-4)*260, 7.73*10**(-4)*300]
}


exp = pd.DataFrame(
    data = exp_data,
    index = dtPoints.keys()    
)

print(exp)

# Dot plot #
# Colour palette
palette = ["royalblue", "green", "red", "magenta"]

# Marker styles
lineStyle = ["-->", ":1", ":<", ":+", "-8"]

# Include the mean values
mean_points = pd.DataFrame({
    "Names": [labels[0], labels[1], labels[2], labels[3]],
    "Values": [meanIdle, meanTO, meanCO, meanApp]
})

# Dot plot
distribution_plots(
    df_all, 
    mean_points, 
    dtCorrs,
    exp,
    method = "Swarmplot",
    size = [10,7],
    ylimits = [0, 50, 10], # min, max, step 
    title = "NOx EI over engine operation points - Dot plot - CFM56 family", 
    xLabel = "Pollutant and operating point", 
    yLabel = "Emissions index value (g/kg)", 
    colours = palette, 
    labels = labels,  
    dotPlotXlabel = "Pollutant", 
    dotPlotYlabel = "Value", 
    lineStyle = lineStyle
)