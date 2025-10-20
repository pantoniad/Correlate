from Polynomial_main import polynomial_main

## Main inputs ##
# Secondary
# Bias
include_bias = False
include_plots = True
save_results = True

# Primary
# Train size, Degree of polynomial
primary_inputs = {
    "Idle": {
        "Train split": 0.6,
        "Include development split": False,
        "Degree of polynomial": 2    
    },
    "T/O":{
        "Train split": 0.6,
        "Include development split": False,
        "Degree of polynomial": 2    
    },
    "C/O": {
        "Train split": 0.6,
        "Include development split": False,
        "Degree of polynomial": 2    
    },
    "App": {
        "Train split": 0.6,
        "Include development split": False,
        "Degree of polynomial": 2    
    }
}

engine_specs = {
    "Pressure Ratio": 27.6,
    "Rated Thrust (kN)": 117,
    "Fuel flow Idle (kg/s)": 0.144,
    "Fuel flow Take-off (kg/s)": 1.25,
    "Fuel flow Climb-out (kg/s)": 1,
    "Fuel flow Approach (kg/s)": 0.348
}

polynomial_main(model_structure = primary_inputs, engine_specs = engine_specs,
                include_bias = include_bias, include_plots = include_plots, 
                save_results = save_results)