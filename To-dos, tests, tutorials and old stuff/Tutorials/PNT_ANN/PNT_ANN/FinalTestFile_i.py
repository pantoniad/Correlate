from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import AssistFunctionsPytorch as assist
import seaborn as sns
import os
from tensorflow import keras

def test(DataSetFile, sheet_name, model_directory, neurons, layers, activ_func, mode, input_features, output_features, approach):
    # paths to the saved model and scaler

    split_up = os.path.splitext(DataSetFile)
    if split_up[1] == '.xlsx':
        test_data = pd.read_excel(DataSetFile, sheet_name=sheet_name)
    else:
        test_data = pd.read_csv(DataSetFile, sep=",")

    X_test = test_data[input_features]
    y_test = test_data[output_features]


    if approach == "PyTorch":
        model_path = f'{model_directory}/best_mlp_model.pth'
        scaler_path = f'{model_directory}/scaler.pkl'

        # Set activation function
        if activ_func == "ReLU":
            activ_func = nn.ReLU()
        else:
            print("Error activation function.")

        input_size = np.shape([input_features])[-1]
        output_size = np.shape([output_features])[-1]
        best_model = assist.load_model(model_path, input_size, output_size, neurons, layers, activ_func, mode)
        scaler = assist.load_scaler(scaler_path)
        X_test_tensor = assist.preprocess_data(X_test.values, scaler)
        predictions = assist.predict(best_model, X_test_tensor)
    elif approach == "TensorFlow":
        predict_model = keras.models.load_model("flightpath_parallel_100K/v2/Energy_Consumption_model/512_neurons_8_layers/TensorFlow/best_mlp_model")
        predictions = predict_model(X_test)
        predictions = np.swapaxes(predictions, 1, 0)[0]
    else:
        print("Error ANN Approach!")
        exit()
    if output_features == "Energy_Consumption":
        predictions_np = np.array(predictions) / 1e+9
        test_data["Energy_Consumption"] = predictions_np.copy()
        actuals_np = y_test.to_numpy() / 1e+9
        metric = "Energy Consumption"
        unit = "GJ"

    else:
        print("Error Output Feature.")
        exit()

    # print("Writing Dataset File to Excel.")
    # test_data["Predictions_ff_partial"] = predictions
    # test = pd.ExcelWriter("For_Michalis/1spAxCf.xlsx")
    # test_data.to_excel(test, index=False)
    # test.close()
    # exit()

    over_under = 100 * np.sum((predictions_np - actuals_np) / actuals_np) / np.shape(predictions_np)[0]
    if over_under > 0:
        print(f"The model over predicts {metric} values by {over_under}%.")
    elif over_under < 0:
        print(f"The model under predicts {metric} values by {over_under}%.")
    else:
        print(f"The model predicts {metric} values with perfect accuracy.")

    test_mape = mean_absolute_percentage_error(actuals_np, predictions_np)
    print(f'Mean Absolute Percentage Error on Test Data: {test_mape * 100}%')

    plt.rcParams.update({"font.size": 24})
    plt.figure(figsize=(12, 7))
    plt.scatter(actuals_np, predictions_np, c='blue', label='Predictions', marker='o', s=20, alpha=0.5)
    plt.title(f'{metric} Prediction against Actual')
    plt.xlabel(f'{metric} [{unit}]')
    plt.ylabel(f'Predicted {metric} [{unit}]')
    plt.plot([min(actuals_np), max(actuals_np)], [min(actuals_np), max(actuals_np)], 'k--',
             label='Ideal Prediction')  # ideal line for reference
    plt.legend()
    plt.grid(True)
    plt.show()

    # individual MAPE for each data point
    individual_mape = np.abs((predictions_np - actuals_np) / actuals_np) * 100
    test_data["MAPE [%]"] = individual_mape

    plt.figure(figsize=(12, 7))
    plt.scatter(range(len(individual_mape)), individual_mape, color='black', marker='o', s=20, alpha=0.3)
    plt.title('MAPE Error for Each Data Point')
    plt.xlabel('Data Point Index')
    plt.ylabel('MAPE Error [%]')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 7))
    plt.scatter((actuals_np), individual_mape, color='black', marker='o', s=10, alpha=0.5)
    plt.title('MAPE Error for Each Data Point')
    plt.xlabel(f'{metric} [{unit}]')
    plt.ylabel('MAPE Error [%]')
    plt.grid(True)
    plt.show()



    # plt.figure(figsize=(12, 7))
    # test_data["Flight_Condition"] = test_data["Flight_Condition"].replace("MTO", "NTO")
    # unique_conditions = test_data["Flight_Condition"].unique()
    # palette = sns.color_palette("Greys", len(unique_conditions))  # Using a colorful palette
    # color_mapping = dict(zip(unique_conditions, palette))
    # sns.boxplot(data=test_data, x="Flight_Condition", y="MAPE [%]", hue="Flight_Condition", palette=palette, dodge=False)
    # plt.legend([], [], frameon=False)
    # plt.grid(True)
    # plt.ylim((-1, 25))
    # # plt.title("MAPE Boxplot per Flight Condition")
    # plt.show()

    # Histogram
    individual_mape = np.abs((predictions_np - actuals_np) / actuals_np) * 100
    # max MAPE for setting the last bin in the histogram
    max_mape = np.max(individual_mape)
    if max_mape > 10:
        mape_bins = [0, 0.5, 1, 2.5, 5, 10, max_mape]
        category_labels = ['< 0.5%', '< 1%', '< 2.5%', '< 5%', '< 10%', f'Max ({max_mape:.2f}%)']
    else:
        mape_bins = [0, 0.5, 1, 2.5, 5, 10]
        category_labels = ['< 0.5%', '< 1%', '< 2.5%', '< 5%', '< 10%']

    mape_counts, _ = np.histogram(individual_mape, bins=mape_bins)
    # percentage of total predictions for each bin-category
    mape_percentages = (mape_counts / len(individual_mape)) * 100
    plt.figure(figsize=(12, 7))
    plt.bar(category_labels, mape_percentages, color='skyblue')
    plt.xlabel('MAPE Categories')
    plt.ylabel('Percentage of Predictions')
    plt.title('Percentage of Predictions by MAPE Categories')
    plt.show()

    # Cumulative MAPE plot for predictions

    plt.figure(figsize=(12, 7))

    total_predictions = len(individual_mape)
    percentages = [
        np.sum(individual_mape <= 0.5) / total_predictions * 100,
        np.sum((individual_mape <= 1)) / total_predictions * 100,
        np.sum((individual_mape <= 2.5)) / total_predictions * 100,
        np.sum((individual_mape <= 5)) / total_predictions * 100,
        np.sum((individual_mape <= 10)) / total_predictions * 100,
        np.sum((individual_mape <= 100)) / total_predictions * 100
    ]

    # MAPE categories for the x-axis
    mape_categories = ['≤ 0.5%', '≤ 1%', '≤ 2.5%', '≤ 5%', '≤ 10%', '≤ 80%']

    # plt.figure(figsize=(12, 7))
    print(percentages)
    plt.plot(mape_categories, percentages, marker='o', linestyle='-', color='black')
    plt.xlabel('MAPE Categories')
    plt.ylabel('Cumulative Percentage of Predictions (%)')
    # plt.title('Cumulative Percentage of Predictions by MAPE')
    # plt.legend(("With Power", "Without Power"))
    plt.grid(True)
    plt.ylim(0, 110)
    plt.show()
