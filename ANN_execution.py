from ANN_main import ann_main

## Main inputs ##
# Secondary inputs
device = "CPU"
include_plots = True
save_results = True

# Main inputs - Structure of ANNs per operating point 
model_str = {
    "Idle": {
        "Learning rate": 1e-3,
        "Epochs": 1500,
        "Number of FC layers": 3,
        "Number of nodes per layer": [3, 2, 2, 1, 1],
        "Activation Function": "relu",
        "Optimizer": "ASDG"
    },
    "T/O":{
        "Learning rate": 1e-3,
        "Epochs": 500,
        "Number of FC layers": 3,
        "Number of nodes per layer": [3, 5, 10, 5, 1],
        "Activation Function": "relu",
        "Optimizer": "Adam"
    },
    "C/O":{
        "Learning rate": 1e-3,
        "Epochs": 500,
        "Number of FC layers": 3,
        "Number of nodes per layer": [3, 5, 10, 5, 1],
        "Activation Function": "relu",
        "Optimizer": "Adam"
    },
    "App":{
        "Learning rate": 1e-3,
        "Epochs": 500,
        "Number of FC layers": 3,
        "Number of nodes per layer": [3, 5, 10, 5, 1],
        "Activation Function": "relu",
        "Optimizer": "Adam"
    }
}

# Initialize model
ann_main(model_structure = model_str, device = device, include_plots = include_plots,
         save_results = save_results)