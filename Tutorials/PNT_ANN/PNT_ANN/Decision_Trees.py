import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.tree import plot_tree
import DT_Assistance as dt_assist
import ANN_Assistance as assist

# class Train
def train(approach, DataSetDirectory, DataSetDoc, train_rate, model_directory, max_depth, min_samples_split, min_samples_leaf,
          criterion, task, seed, input_features, output_features):
    # REFINE OUTPUT FEATURES WITH APPROPRIATE FORMAT AT HACKATHON

    if output_features == "Energy_Consumption":
        output_features_tag = "Energy_Consumption"
        X_DataSetFile = f"{DataSetDirectory}/input_ann_sorted_data.txt"
        y_DataSetFile = f"{DataSetDirectory}/obj_func_ann.txt"
        split_up_X = os.path.splitext(X_DataSetFile)
        split_up_y = os.path.splitext(y_DataSetFile)

        if split_up_X[1] == '.xlsx':
            X_DataSet = pd.read_excel(X_DataSetFile, header=None, names=input_features)
        else:
            X_DataSet = pd.read_csv(X_DataSetFile, delim_whitespace=True, header=None, names=input_features)

        if split_up_y[1] == '.xlsx':
            y_DataSet = pd.read_excel(y_DataSetFile, header=None, names=[output_features])
        else:
            y_DataSet = pd.read_csv(y_DataSetFile, delim_whitespace=True, header=None, names=[output_features])

    elif output_features == "Power[W]":
        output_features_tag = "Power"
        DataSetFile = f"{DataSetDirectory}/Power_Dataset.txt"
        split_up = os.path.splitext(DataSetFile)
        if split_up[1] == '.xlsx':
            DataSet = pd.read_excel(DataSetFile)
        else:
            DataSet = pd.read_csv(DataSetFile, delim_whitespace=True)
        X_DataSet = DataSet[input_features]
        y_DataSet_temp = DataSet[output_features]
        y_DataSet = y_DataSet_temp.to_frame()

  
    else:
        print("Error Output Feature(s).")
        exit()

    if model_directory is None:
        model_directory = DataSetDirectory
    else:
        model_directory = f"{DataSetDirectory}/{model_directory}"

    # Create model directory
    if not os.path.exists(
            f"{model_directory}/{output_features_tag}_model/{max_depth}_depth_{min_samples_split}_split_{task}/{approach}"):
        os.makedirs(
            f"{model_directory}/{output_features_tag}_model/{max_depth}_depth_{min_samples_split}_split_{task}/{approach}")

    # Set model mode and sort data accordingly
    np.random.seed(seed)

    X = X_DataSet[input_features]
    y = y_DataSet[output_features]

    # Split the data

    X_train_temp, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - train_rate), random_state=seed)

    X_train = X_train_temp[input_features]

    if not os.path.exists("test_data"):
        os.makedirs("test_data")

    # save test data in model directory
    test = pd.ExcelWriter(f"{model_directory}/{output_features_tag}_model/test_2025_2_16_10.xlsx")
    test_data = pd.concat([X_test, y_test], axis=1)  # Combine features and target variable
    test_data.to_excel(test, index=False)
    test.close()

    # save input features in model directory
    input_features_panda = pd.DataFrame(input_features, columns=["Input Features"])
    input_features_panda.to_csv(
        f"{model_directory}/{output_features_tag}_model/{max_depth}_depth_{min_samples_split}_split_{task}/{approach}/input_features.txt",
        sep=" ")

    # save model parameters
    model_parameter_values = [approach, train_rate, max_depth, min_samples_split, min_samples_leaf, seed]
    model_parameter_tags = ["approach", "train_rate", "max_depth", "min_samples_split", "min_samples_leaf", "seed"]
    model_parameters = pd.DataFrame([model_parameter_values], columns=model_parameter_tags)
    model_parameters.to_csv(
        f"{model_directory}/{output_features_tag}_model/{max_depth}_depth_{min_samples_split}_split_{task}/{approach}/model_parameters.txt",
        sep=" ")

    if approach == "Random_Forest":
        model_param = dt_assist.train_random_forest(max_depth, min_samples_split, min_samples_leaf, seed, criterion,
                                                    task)
        model = model_param[0]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        loss_function = model_param[1]
        acc = loss_function(y_test, y_pred)
        loss_tag = model_param[2]

        print(
            f"Random Forest with max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
        print(f"{loss_tag}: {acc:.4f}\n")



    elif approach == "Decision_Trees":
        model_param = dt_assist.train_decision_tree(max_depth, min_samples_split, min_samples_leaf, seed, criterion, task)
        model = model_param[0]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        loss_function = model_param[1]
        acc = loss_function(y_test, y_pred)
        loss_tag = model_param[2]

        print(
            f"Decision Tree with max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
        print(f"{loss_tag}: {acc:.4f}\n")


    # elif approach == "Gradient_Boost":

    else:
        print("Error decision tree method!")
        exit()

    if task == "Regression":
        plt.figure(figsize=(12, 6))
        plot_tree(model, feature_names=input_features, filled=True)
        plt.show()
    elif task == "Classification":
        plt.figure(figsize=(12, 6))
        plot_tree(model, feature_names=input_features, class_names=[output_features], filled=True)
        plt.show()
    else:
        print("Error method!")
        exit()
    plt.savefig('books_read.png')

    joblib.dump(model, "best_model.pkl")
    print("Model saved successfully!")