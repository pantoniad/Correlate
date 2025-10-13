from GradientBoosting_main import gbr_main

## Secondary inputs
include_learning_curve = False
include_complexity_plot = False
save_results = True

## Primary inputs
primary_inputs = {
    "Idle": {
        "Train split": 0.5,
        "Include development split": False,
        "Number of estimators": 3500,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 1,
        "Subsample size": 0.4
    },
    "T/O":{
        "Train split": 0.5,
        "Include development split": False,
        "Number of estimators": 3500,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 1,
        "Subsample size": 0.4
    },
    "C/O": {
        "Train split": 0.5,
        "Include development split": False,
        "Number of estimators": 1250,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 150,
        "Subsample size": 0.4
    },
    "App": {
        "Train split": 0.5,
        "Include development split": False,
        "Number of estimators": 4000,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 1,
        "Subsample size": 0.4
    }
}

gbr_main(model_structure = primary_inputs, include_learning_curve = include_learning_curve,
         include_complexity_plot = include_complexity_plot, save_results = save_results)