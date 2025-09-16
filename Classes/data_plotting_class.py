import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt  
import matplotlib.gridspec as gridspec  
import Classes.correlations_class as correlate
import warnings
from Classes.latex_class import latex as lx
from Classes.FuelFlow_class import FuelFlowMethods as ffms

class data_plotting:


    def __init__(self, mean_points: pd.DataFrame,
                 dtCorrs: pd.DataFrame,
                 exp: pd.DataFrame
                ):
        """
        Inputs:
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
        """
        self.mean_points = mean_points
        self.dtCorrs = dtCorrs
        self.exp = exp
        
    def distribution_plots(self, df_all: pd.DataFrame,
                           method: str, 
                           size: list, 
                           title: str, 
                           ylimits: list, 
                           xLabel: str, 
                           yLabel: str, 
                           colours: list, 
                           labels: list, 
                           dotPlotXlabel: str, 
                           dotPlotYlabel: str, 
                           lineStyle: list, 
                           Jitter :float = None):
        """
        distribution_plots: a function that is able to generate four kinds of plots based on the 
                            string given in the "method" parameter of the function. 
                            Supported DISTRIBUTION plots: Dot plots, Box plots, Violin Plots, Swarm plots

        Inputs:
        - self: -
        - df_all:   data to be placed as dots in dot plot
                    Dataframe, first column are the names
                    for each data position, second column 
                    are the values
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

        # Extract data from self
        mean_points = self.mean_points
        dtCorrs = self.dtCorrs
        exp = self.exp    
        
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

    def error(self):
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
        - self: - 
        
        Outputs:
        - meanRelativeEC: the mean relative error between all of the correlation
                            equations and the mean values of each operating point, 
                            for all operating points, Dataframe 
        - meanRelativeEE: same with the above, but for the experimental datasets, 
                            Dataframe
        """

        # Extract data from self
        dtCorrs = self.dtCorrs
        mean_points = self.mean_points
        exp = self.exp
        
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
    
    def plot_3d(self):
    
        pass
