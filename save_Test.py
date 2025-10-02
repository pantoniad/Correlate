from Classes.data_processor_class import data_process
import pandas as pd
## Main inputs ##
# Secondary inputs
device = "GPU"
include_plots = True
save_results = True

# Main inputs - Structure of ANNs per operating point 
model_str = {
    "Idle": {
        "Learning rate": 1e-3,
        "Epochs": 500,
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

secondary_inputs = pd.DataFrame(
    data = {
        "Device": device,
        "Include plots": include_plots,
        "Save results": save_results
    },
    index = ["Value"]
)

model_str = pd.DataFrame(
    data = model_str.pivot(index = model_)
)

data_process.data_saver(model_str, secondary_inputs, model = "Polynomial Regression")