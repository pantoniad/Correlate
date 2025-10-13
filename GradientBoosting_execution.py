from GradientBoosting_main import gbr_main

## Secondary inputs
include_learning_curve = True
include_complexity_plot = True
save_results = True

## Primary inputs
primary_inputs = {
    "Idle": {
        "Train split": 0.7,
        "Include development split": False,
        "Number of estimators": 200,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 40,
        "Subsample size": 0.4
    },
    "T/O":{
        "Train split": 0.7,
        "Include development split": False,
        "Number of estimators": 200,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 150,
        "Subsample size": 0.4
    },
    "C/O": {
        "Train split": 0.7,
        "Include development split": False,
        "Number of estimators": 200,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 150,
        "Subsample size": 0.4
    },
    "App": {
        "Train split": 0.7,
        "Include development split": False,
        "Number of estimators": 140,
        "Learning rate": 1e-3,
        "Criterion":  "friedman_mse",
        "Maximum Tree depth": 80,
        "Subsample size": 0.4
    }
}

gbr_main(model_structure = primary_inputs, include_learning_curve = include_learning_curve,
         include_complexity_plot = include_complexity_plot, save_results = save_results)