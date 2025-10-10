from GradientBoosting_main import gbr_main

## Secondary inputs
include_plots = True
save_results = True

## Primary inputs
primary_inputs = {
    "Idle": {
        "Train split": 0.7,
        "Include development split": False,
        "Number of estimators": 200,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 10
    },
    "T/O":{
        "Train split": 0.7,
        "Include development split": False,
        "Number of estimators": 200,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 10
    },
    "C/O": {
        "Train split": 0.7,
        "Include development split": False,
        "Number of estimators": 200,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 10
    },
    "App": {
        "Train split": 0.7,
        "Include development split": False,
        "Number of estimators": 200,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 10
    }
}

gbr_main(model_structure = primary_inputs, include_plots = include_plots, 
         save_results = save_results)