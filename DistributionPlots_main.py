# Import
import pandas as pd
import numpy as np
import copy
import os

from Classes.data_plotting_class import data_plotting
from Classes.data_processor_class import data_process
from Classes.models_class import models_per_OP
from Classes.correlations_class import Correlations
from Classes.FuelFlow_class import FuelFlowMethods
from Classes.latex_class import latex

def DistributionPlots_main(operating_conditions: dict, thermodynamic: dict, engine_specs: dict,
                           correlation_equations: dict, fuel_flow_method: dict, experimental_data: dict, icao_data: dict,
                           surrogate_models: dict, distribution_plot_settings: dict, save_plots: dict = False,
                           save_results: bool = True, notes: str = "None"):
    """
    DistributionPlots_main: A function created in conjuction with DistributionPlots_execution
    to handle the creation of distribution plots. The data used are retrieved directly from the 
    ICAO Emissions Databank

    Inputs:
    - operating_conditions: dictionary that contains the operating conditions considered for the
    engine. Must be included. 
    Structure: dictionary: {"Export LaTeX table": boolean, "Flight altitude (m)": [], 
    "Flight speed (Mach number)": [], "Axial fan speed (Mach number)": []},
    Parameter list structure (i.e. Flight altitude): ["Idle", "Take-off", "Climb-out", "Approach"]

    - thermodynamic: dictionary that contains the values taken from AeroEngineS for each operating
    point. Must be included. 
    Structure: dictionary: {"Export LaTeX table": Boolean, "Idle": [], "Take-off": [],
    "Climb-out": [], "Approach": []}, 
    Operating point list structure: [Combustor Inlet Temperature (K), Combustor Outlet Temperature (K),
    Combustor Inlet Pressure (Pa), Core air mass flow rate (kg/s), Fuel flow rate (kg/s)]

    - engine_specs: dictionary that contains the specifications of the engine under investigation.
    Structure: {"Include": boolean, "Export LaTeX table": boolean, "Rated Thrust (kN)": [], "Fan diameter": [],
    "Hub2Tip ratio": [], "Bypass ratio": [], "Fan Pressure ratio": [], "Booster Pressure ratio": [], 
    "High Pressure compressor Pressure ratio": [], "Overall Pressure ratio": []}

    - correlation_equations: dictionary that contains the names of the correlation equations to be used,
    as keys, and a boolean value as the value.
    Structure: {"Include": boolean, "Export LaTeX table": boolean, "<Name of the correlation>": boolean}

    - fuel_flow_method: dictionary that enables the usage of fuel flow methods (DLR) and specifies the 
    engine that the method should be used for,
    Structure: {"Include": boolean, "Export LaTeX table": boolean, "Engine model": str}

    - experimental_data: multi-layer dictionary that contains, in the first layer, the source of the
    experimental data and, on the second layer, contains information about that data.
    Structure: {"Include": boolean, "<Source of the data>":{
    "Include": Boolean, "Export LaTeX table": boolean, "<Operating point> EI (g/kg): float}}
    
    - icao_data: dictionary that handles the usage of ICAO data. The columns considered are included in 
    the main file and the user does not need to define them. Structure: {"Include": boolean, 
    "File path": <Path to the file>, "Row range for engine family": nested list, 
    "Row range for specific engine from the family": nested list}

    - surrogate_models: multi-layer dictionary that contains the models to be used and the directory that 
    the predicted value for the EI of the engine under investigation is stored. The stored file must be a csv.
    Structure: {"Include": boolean, "<Model>": {
    "Include": boolean, "Path to predicted engine EI": str}}

    - distribution_plot_settings: dictionary that contains the parameters used for the formation of the
    distribution plot. Must be given.
    Structure: {"Plot type": "Violinplot", "Pollutant type": str, "Title": str, "X-axis label": str,
    "Y-axis label": str}

    Outputs:
    - Plots save to the plots_save_path
    
    """
    ## Operating points
    op = ["Idle", "T/O", "C/O", "App"] # For general usage
    labels = ["Idle", "Take-off", "Climb-out", "Approach"] # For plotting
    
    ## Data unpacking
    thermodynamic_df = pd.DataFrame(
        data = thermodynamic,
        index = thermodynamic["Index"]
    ).drop(["Export LaTeX table", "Index"], axis = 1)
    thermodynamic_df = thermodynamic_df.rename(columns = {"Take-off": op[1], "Climb-out": op[2], "Approach": op[3]})

    pollutant = distribution_plot_settings["Pollutant to consider"]

    ## Data and plots saving ##
    # Convert inputs to dataframes
    dt_secondary = {
        "Save generated plots": save_plots,
        "Save arithmetic results": save_results
    }
    secondary = pd.DataFrame(data = dt_secondary, index = ["Argument"])

    surrogate_models_primary = pd.DataFrame(data = surrogate_models).T
    correlation_equations_primary = pd.DataFrame(data = correlation_equations, index = ["Argument/Value"])
    thermodynamic_primary = pd.DataFrame(data = thermodynamic).T
    engine_specs_primary = pd.DataFrame(data = engine_specs, index = ["Argument/Value"]).T
    operating_conditions_primary = pd.DataFrame(operating_conditions, index = labels)
    fuel_flow_method_primary = pd.DataFrame(fuel_flow_method, index = ["Argument/Value"]).T
    experimental_data_primary = pd.DataFrame(experimental_data)
    plot_settings_primary = pd.DataFrame(distribution_plot_settings, index = ["Argument/Value"]).T

    # Get paths for plots and results saving
    if save_results == True and save_plots == True:

        error_save_path, plots_save_path = data_process.data_saver_distribution(
            secondary_inputs = secondary, surrogate_models = surrogate_models_primary, correlations_equations = correlation_equations_primary,
            thermodynamic = thermodynamic_primary, engine_specs = engine_specs_primary, operating_conditions = operating_conditions_primary, 
            fuel_flow_method = fuel_flow_method_primary, experimental_data = experimental_data_primary, plot_settings = plot_settings_primary,
            notes = notes    
        )

    elif save_results == True:

        error_save_path, plots_save_path = data_process.data_saver_distribution(
            secondary_inputs = secondary, surrogate_models = surrogate_models_primary, correlations_equations = correlation_equations_primary,
            thermodynamic = thermodynamic_primary, engine_specs = engine_specs_primary, operating_conditions = operating_conditions_primary, 
            fuel_flow_method = fuel_flow_method_primary, experimental_data = experimental_data_primary, plot_settings = plot_settings_primary,
            notes = notes    
        )

        plots_save_path = None

    elif save_plots == True:

        error_save_path, plots_save_path = data_process.data_saver_distribution(
            secondary_inputs = secondary, surrogate_models = surrogate_models_primary, correlations_equations = correlation_equations_primary,
            thermodynamic = thermodynamic_primary, engine_specs = engine_specs_primary, operating_conditions = operating_conditions_primary, 
            fuel_flow_method = fuel_flow_method_primary, experimental_data = experimental_data_primary, plot_settings = plot_settings_primary,
            notes = notes    
        )

        plots_save_path = None

    else:

        error_save_path, plots_save_path = None, None

    ## ICAO data ##
    if icao_data["Include"]:

        # Extract data
        path = icao_data["File path"]
        df = pd.read_csv(path, delimiter = ";")

        clmns = ["Pressure Ratio", "Rated Thrust (kN)", f"Fuel Flow {op[0]} (kg/sec)", 
                f"Fuel Flow {op[1]} (kg/sec)", f"Fuel Flow {op[2]} (kg/sec)", f"Fuel Flow {op[3]} (kg/sec)",
                f"{pollutant} EI {op[0]} (g/kg)", f"{pollutant} EI {op[1]} (g/kg)", f"{pollutant} EI {op[2]} (g/kg)",
                f"{pollutant} EI {op[3]} (g/kg)"]

        drange = icao_data["Row range for engine family"] 

        df = data_process.csv_cleanup(df = df, clmns = clmns, drange = drange, reset_index = True, save_to_csv = True, path = r"Databank/EngineData.csv")

        # Create the df_all for the distribution_plots method
        dfa = df.filter(df.columns[df.columns.str.contains(pollutant)], axis = 1)

        plots = []
        for i in op:
            dfb = dfa.filter(df.columns[df.columns.str.contains(i)], axis = 1)

            dfb_values = dfb[f"{pollutant} EI {i} (g/kg)"].astype(float).values
            plots.append((f"{pollutant} {i}", np.sort(dfb_values)))

        icao_points = pd.DataFrame({
            "Pollutant": np.concatenate([[p[0]] * len(p[1]) for p in plots]),
            "Value": np.concatenate([p[1] for p in plots])
        })

        # Get mean values per operating point
        meanIdle = np.mean(df[f"{pollutant} EI Idle (g/kg)"].values.astype(float))
        stdIdle = np.round(np.std(df[f"{pollutant} EI Idle (g/kg)"].values.astype(float)),3)

        meanTO = np.mean(df[f"{pollutant} EI T/O (g/kg)"].values.astype(float))
        stdTO = np.round(np.std(df[f"{pollutant} EI T/O (g/kg)"].values.astype(float)),3)

        meanCO = np.mean(df[f"{pollutant} EI C/O (g/kg)"].values.astype(float))
        stdCO = np.round(np.std(df[f"{pollutant} EI C/O (g/kg)"].values.astype(float)),3)

        meanApp = np.mean(df[f"{pollutant} EI App (g/kg)"].values.astype(float))
        stdApp = np.round(np.std(df[f"{pollutant} EI App (g/kg)"].values.astype(float)),3)

        mean_points = pd.DataFrame(
            data = {
            pollutant: [meanIdle, meanTO, meanCO, meanApp]
            },
            index = labels
        )

        # Get data for specific engine
        engineIdle = df.iloc[74:75]["NOx EI Idle (g/kg)"].values[0]
        engineTO = df.iloc[74:75]["NOx EI T/O (g/kg)"].values[0]
        engineCO = df.iloc[74:75]["NOx EI C/O (g/kg)"].values[0]
        engineApp = df.iloc[74:75]["NOx EI App (g/kg)"].values[0]

        engine_icao_eis = pd.DataFrame(
            data = {"EI": [engineIdle, engineTO, engineCO, engineApp]},
            index = labels  
        ).T

    else:

        icao_points = pd.DataFrame([])
        mean_points = pd.DataFrame([])
        engine_icao_eis = pd.DataFrame([])

    ## Experimental data ##
    if experimental_data["Include"] == True:
        
        # Data in dataframe 
        exp_data_dt = experimental_data["Turgut - CFM56/7B26"]
        
        exp_data = pd.DataFrame(
            data = exp_data_dt,
            index = ["Turgut - CFM56/7B26"]
        ).drop(["Include", "Export LaTeX table"], axis = 1).T
        
        exp_data = exp_data.rename(index={"Idle EI (g/kg)": op[0], "Take-off EI (g/kg)": op[1],
                               "Climb-out EI (g/kg)": op[2], "Approach EI (g/kg)": op[3]})

        # Export to LaTeX table
        #if experimental_data["Turgut - CFM56/7B26"]["Export LaTeX table"]:
            
            #exp_table = latex(df = exp_data, filename = "", caption = "", label = "")

    else:

        exp_data = pd.DataFrame([])

    ## Correlation equations ##
    if correlation_equations["Include"]:
        
        # Empty dataframe for storage
        dtCorrs = pd.DataFrame([])
        d = {}
        corrs = copy.deepcopy(correlation_equations)
        
        # Correlations instance
        for i in op:

            # Create class instance for opeating point
            corr = Correlations(
                thermodynamic_df[i]["Tbin"], 
                thermodynamic_df[i]["Tbout"], 
                thermodynamic_df[i]["Pbin"], 
                0.95*thermodynamic_df[i]["Pbin"], 
                thermodynamic_df[i]["FAR"], 
                thermodynamic_df[i]["m_dot_air"], 
                0, 
                1.293
            )

            # Get values from correlation equations
            if correlation_equations["Becker"]:
                becker = corr.becker(1800, method = "simplified")
                d["Becker"] = becker

            if correlation_equations["Rokke"]:
                rokke = corr.rokke_nox(engine_specs["Overall Pressure ratio"], method = "Index")
                d["Rokke"] = rokke
            if correlation_equations["Lewis"]:
                lewis = corr.lewis_nox()
                d["Lewis"] = lewis
            
            if correlation_equations["Kyprianidis"]:
                kyprianidis = corr.kyprianidis(h = 0)
                d["Kyprianidis"] = kyprianidis
            
            if correlation_equations["Novelo"]:
                novelo = corr.novelo()
                d["Novelo"] = novelo
            
            if correlation_equations["Perkavec"]:
                perkavec = corr.perkavec()
                d["Perkavec"] = perkavec
            
            if correlation_equations["Lefebvre"]:
                lefebvre = corr.lefebvre(Vc = 0.05, Tpz = 2000, Tst = 2250)
                d["Lefebvre"] = lefebvre
            
            if correlation_equations["GasTurb"]: 
                gasturb = corr.gasturb(WAR = 0)
                d["GasTurb"] = gasturb
            
            if correlation_equations["General Electric"]:
                ge = corr.generalElectric(WAR = 0)
                d["General Electric"] = ge
            
            if correlation_equations["Aeronox"]:
                aeronox = corr.aeronox(Vc = 0.1, R = 287)
                d["Aeronox"] = aeronox
  
            index = [i]

            dt1 = pd.DataFrame(
                data = d,
                index = index
            )

            # Append temporary dataframe to external
            dtCorrs = pd.concat([dtCorrs, dt1], axis = 0)
    
    else: # No correlation equation is used
        data = {"Idle": 0, "T/O": 0, "C/O": 0, "App": 0}
        dtCorrs = pd.DataFrame(data, index = ["Point"]).T

    ## Fuel flow methods ##
    if fuel_flow_method["Include"]:

        engine_specific_range = fuel_flow_method["ICAO Emissions Databank range"]
        df_ffm = copy.deepcopy(df)
        df_ffm = df_ffm.iloc[range(engine_specific_range[0][0], engine_specific_range[0][1])].reset_index()
        df_ffm = df_ffm.drop(["index","Pressure Ratio", "Rated Thrust (kN)"], axis = 1)
        
        # Operating conditions 
        speed = operating_conditions["Flight speed (Mach number)"]  
        alt = operating_conditions["Flight altitude (m)"]     
        
        d_ffm= {
            "EINOx": df_ffm.iloc[0][4:9].values.astype(float),
            "Fuel Flows": df_ffm.iloc[0][0:4].values.astype(float),
            "Flight altitude": alt,
            "Flight Speed": speed
        }

        datapoints = pd.DataFrame(
            data = d_ffm,
            index = ["Idle", "Take-off", "Climb-out", "Approach"]
        )
        datapoints = datapoints.T

        ff = FuelFlowMethods(datapoints = datapoints, fitting = "Parabolic", check_fit = False)
        pred_ei_ff = ff.dlrFF()

        # Add fuel flow EIs to dtCorrs
        dtCorrs["DLR Fuel Flow"] = pred_ei_ff.values.T

        if dtCorrs.keys()[0] == "Point":
            dtCorrs = dtCorrs.drop("Point", axis = 1)
    
    else:

        dtCorrs = pd.DataFrame([])
    
    ## Surrogate models ##
    if surrogate_models["Include"]:
        
        # Input dataframe
        dt_surrogates = copy.deepcopy(surrogate_models)
        del dt_surrogates["Include"]
        df_surrogates_inputs = pd.DataFrame(data = dt_surrogates).T

        # Output dataframe
        df_surrogates_preds = pd.DataFrame([])

        for i in op:
            
            # Intermediate variable to handle paths
            if i == "T/O":
                oper = "TO"
            elif i == "C/O":
                oper = "CO"
            else:
                oper = i

            # Define intermediate dataframe
            df_inter = pd.DataFrame([])

            # Iterate through the models
            for (j, index) in enumerate(df_surrogates_inputs.index):
                
                if  df_surrogates_inputs.values[j][0]:
                    
                    # Retrieve data
                    surrogate_pred = pd.read_csv(os.path.join(df_surrogates_inputs.values[j][1], f"engine_EI_pred_{oper}.csv"))
                    surrogate_pred = surrogate_pred.drop(["Unnamed: 0", "Engine model", "Pressure ratio", "Rated thrust (kN)", "Fuel flow (kg/s)"], axis = 1)
                    surrogate_pred = surrogate_pred.rename(index = {0 : index})
                    surrogate_pred = surrogate_pred.T
                    surrogate_pred = surrogate_pred.rename({"Predicted EI value (gNOx/kgFuel)": i})

                # Concatinate along the x axis
                df_inter = pd.concat([df_inter, surrogate_pred], axis = 1)
            
            # Concatinate along the y-axis
            df_surrogates_preds = pd.concat([df_surrogates_preds, df_inter], axis = 0)

    else:
        
        df_surrogates_preds = pd.DataFrame([])

    ## Distribution plots ##
    
    # Colors and marker styles
    labels_for_plotting = ["NOx Idle", "NOx T/O", "NOx C/O", "NOx App"]
    palette = ["royalblue", "green", "red", "magenta"]
    lineStyle = ["-->", ":1", ":<", ":+", "-8"]

    # Unpack settings
    title = distribution_plot_settings["Pollutant to consider"] + " " + distribution_plot_settings["Title"] + " - " + distribution_plot_settings["Plot type"] + " - " + "CFM56 Family"
    xlabel = distribution_plot_settings["X-axis label"]
    ylabel = distribution_plot_settings["Y-axis label"]

    # Distribution plots
    distr_plots = data_plotting(df_all = icao_points, dtCorrs = dtCorrs, exp = exp_data, mean_points = mean_points, engine_icao_eis = engine_icao_eis, dtmodels = df_surrogates_preds)
    
    if save_plots == False:

        distr_plots.distribution_plots(
            method = distribution_plot_settings["Plot type"],
            size = [12,9],
            title = title,
            xLabel = xlabel,
            yLabel = ylabel,
            colours = palette, 
            labels = labels_for_plotting,  
            dotPlotXlabel = "Pollutant", 
            dotPlotYlabel = "Value", 
            lineStyle = lineStyle 
        )

    else:

        distr_plots.distribution_plots(
            method = distribution_plot_settings["Plot type"],
            size = [12,9],
            title = title,
            xLabel = xlabel,
            yLabel = ylabel,
            colours = palette, 
            labels = labels_for_plotting,  
            dotPlotXlabel = "Pollutant", 
            dotPlotYlabel = "Value", 
            lineStyle = lineStyle,
            save_plots_path = plots_save_path 
        )



