from ANN_main import ann_main

## Model parameters ##
# Secondary inputs
device = "cpu"
include_plots = True
include_complexity_plots = False
save_results = True

# Main inputs - Structure of ANNs per operating point 
model_str = {
    "Idle": {
        "Training split": 0.6,
        "Include development split": False, 
        "Learning rate": 1e-3,
        "Epochs": 200,
        "Number of FC layers": 3,
        "Number of nodes per layer": [3, 5, 5, 5, 1],
        "Activation Function": "relu",
        "Optimizer": "Adam"
    },
    "T/O":{
        "Training split": 0.6,
        "Include development split": False, 
        "Learning rate": 1e-3,
        "Epochs": 200,
        "Number of FC layers": 3,
        "Number of nodes per layer": [3, 5, 5, 5, 1],
        "Activation Function": "relu",
        "Optimizer": "Adam"
    },
    "C/O":{
        "Training split": 0.6,
        "Include development split": False,
        "Learning rate": 1e-3,
        "Epochs": 200,
        "Number of FC layers": 2,
        "Number of nodes per layer": [3, 10, 10, 1],
        "Activation Function": "relu",
        "Optimizer": "Adam"
    },
    "App":{
        "Training split": 0.6,
        "Include development split": False, 
        "Learning rate": 1e-3,
        "Epochs": 200,
        "Number of FC layers": 2,
        "Number of nodes per layer": [3, 5, 5, 1],
        "Activation Function": "relu",
        "Optimizer": "Adam"
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

# Initialize model
ann_main(model_structure = model_str, engine_specs = engine_specs, device = device, include_plots = include_plots,
         include_complexity_plots = include_complexity_plots, save_results = save_results)