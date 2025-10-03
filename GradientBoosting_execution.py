from GradientBoosting_main import gbr_main

## Secondary inputs
include_plots = True
save_results = True

## Primary inputs
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

gbr_main(model_structure = primary_inputs, include_plots = include_plots, 
         save_results = save_results)