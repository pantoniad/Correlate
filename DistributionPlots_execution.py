# Import 
from DistributionPlots_main import DistributionPlots_main
import pandas as pd

## Primary inputs ##
# Engine related
operating_conditions = {
    "Export LaTeX table": False,
    "Flight altitude (m)": [0, 11, 305, 914],          # Idle, Take-off, Climb-out, Approach
    "Flight speed (Mach number)": [0, 0.3, 0.3, 0.2],
    "Axial fan speed (Mach number)": [0.09, 0.4, 0.4, 0.3]
}

thermodynamic = {
    "Export LaTeX table": False,
    "Index": ["Tbin", "Tbout", "Pbin", "m_dot_air", "FAR", "m_dot_fuel"],
    "Idle": [797.1, 1290, 2755850, 10.564, 0.0137, 0.144],      # 
    "Take-off": [809.95, 2250, 2929690, 46.897, 0.0446, 1.25],
    "Climb-out": [805.1, 2000, 2828980, 45.44, 0.03596, 1],
    "Approach": [787.91, 1400, 2539920, 31.89, 0.01718, 0.348]
}

engine_specs = {
    "Include": True,
    "Export LaTeX table": True,
    "Rated Thrust (kN)": [117],
    "Fan diameter": [1.55],
    "Hub2Tip ratio": [0.3],
    "Bypass ratio": [5.1],
    "Fan Pressure ratio": [1.6],
    "Booster Pressure ratio": [1.55],
    "High Pressure compressor Pressure ratio": [11],
    "Overall Pressure ratio": [27.6]
}

# Correlation equations and Semi-empirical methods
correlation_equations = {
    "Include": True,
    "Export LaTeX table": True,
    "Becker": False,
    "Perkavec": False,
    "Rokke": True,
    "Lewis": False,
    "Kyprianidis": True,
    "Novelo": False,
    "Lefebvre": False,
    "GasTurb": False,
    "General Electric": False,
    "Aeronox": False
}

fuel_flow_method = {
    "Include": True,
    "Export LaTeX table": True,
    "Engine model": "CFM56-7B26",
    "ICAO Emissions Databank range": [[74,75]]
}

# Experimental data
experimental_data = {
    "Include": True,
    "Turgut - CFM56/7B26": {
        "Include": True,
        "Export LaTeX table": True,
        "Idle EI (g/kg)": 1.8,
        "Take-off EI (g/kg)": 24.4,
        "Climb-out EI (g/kg)": (12.6+16.4)/2,
        "Approach EI (g/kg)": 2.8
    }
}   

# ICAO data
icao_data = {
    "Include": True,
    "File path": r"Databank/ICAO_data.csv",
    "Row range for engine family": [[61, 169]],
    "Row range for specific engine from the family": [[74, 75]],
    "Include specific engine values on plots": False
}

# Surrogate models
surrogate_models ={
    "Include": True,
    "Polynomial Regression": {
        "Include": True,
        "Path to predicted engine EI": r"E:\Correlate\model_outputs\Run_2025-12-31\Polynomial Regression\ExecutionTime_14-58-35\Polynomial Regression results"
    },
    "Gradient Boosting": {
        "Include": True,
        "Path to predicted engine EI": r"E:\Correlate\model_outputs\Run_2025-12-31\Gradient Boosting\ExecutionTime_14-58-55\Gradient Boosting results\Engine Related Predictions"
    },
    "ANN": {
        "Include": True,
        "Path to predicted engine EI": r"E:\Correlate\model_outputs\Run_2025-12-31\ANN\ExecutionTime_14-59-18\ANN results\Engine Related Predictions"
    }
}

## Secondary inputs ##
distribution_plot_settings = {
    "Plot type": "Violinplot",
    "Pollutant to consider": "NOx",
    "Title": "EI over engine operating points",
    "X-axis label": "Pollutant over operating point",
    "Y-axis label": "Emissions Index Value (g Pollutant/kg Fuel)"
}

save_plots = True
save_results = True

notes_on_the_run = "Distribution plots generated as part of my thesis."

# Run 
DistributionPlots_main(operating_conditions = operating_conditions, thermodynamic = thermodynamic,
                       engine_specs = engine_specs, correlation_equations = correlation_equations,
                       fuel_flow_method = fuel_flow_method, experimental_data = experimental_data, icao_data = icao_data,
                       surrogate_models = surrogate_models, distribution_plot_settings = distribution_plot_settings,
                       save_plots = save_plots, save_results = save_results, notes = notes_on_the_run)
