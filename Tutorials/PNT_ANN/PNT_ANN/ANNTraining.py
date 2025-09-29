import pandas as pd
import AssistFunctionsPytorch as assist
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import joblib
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
from tensorflow.keras.initializers import HeNormal


def train(approach, mode, DataSetDirectory, train_rate, test_rate, model_directory, num_neurons, num_layers,
          activ_func, learning_rate, reg, dropout, loss_func, train_batch_size, val_batch_size, patience,
          training_epochs, seed, input_features, output_features, best_training_epoch, show_progress):

    if output_features == "Energy_Consumption":
        if DataSetDirectory == "80K_dataset":
            print(2)
            output_features_tag = "Energy_Consumption"
            X_DataSetFile = f"{DataSetDirectory}/input_ann_sorted_data.txt"
            y_DataSetFile = f"{DataSetDirectory}/obj_func_ann.txt"
        elif DataSetDirectory == "flightpath_parallel_100K":
            print(1)
            output_features_tag = "Energy_Consumption"
            X_DataSetFile = f"{DataSetDirectory}/corrected_output.csv"
            y_DataSetFile = f"{DataSetDirectory}/obj_func.csv"
        else:
            print("Unknown dataset directory!")
            exit()
    else:
        print("Error Output Feature(s).")
        exit()

    split_up_X = os.path.splitext(X_DataSetFile)
    split_up_y = os.path.splitext(y_DataSetFile)

    if split_up_X[1] == '.xlsx':
        X_DataSet = pd.read_excel(X_DataSetFile, header=None, names=input_features)
    elif split_up_X[1] == '.csv':
        X_DataSet = pd.read_csv(X_DataSetFile, sep=",", index_col=None, header=None, names=input_features)

    else:
        X_DataSet = pd.read_csv(X_DataSetFile, delim_whitespace=True, header=None, names=input_features)

    if split_up_y[1] == '.xlsx':
        y_DataSet = pd.read_excel(y_DataSetFile, header=None, names=[output_features])
    elif split_up_X[1] == '.csv':
        y_DataSet = pd.read_csv(y_DataSetFile, sep=",", index_col=None, header=None, names=[output_features])
    else:
        y_DataSet = pd.read_csv(y_DataSetFile, delim_whitespace=True, header=None, names=[output_features])

    if model_directory is None:
        model_directory = DataSetDirectory
    else:
        model_directory = f"{DataSetDirectory}/{model_directory}"

    # Create model directory
    if not os.path.exists(f"{model_directory}/{output_features_tag}_model/{num_neurons}_neurons_{num_layers}_layers/{approach}"):
        os.makedirs(f"{model_directory}/{output_features_tag}_model/{num_neurons}_neurons_{num_layers}_layers/{approach}")

    # Set model mode and sort data accordingly
    np.random.seed(seed)

    X = X_DataSet[input_features]
    y = y_DataSet[output_features]

    y_true = np.where(y>10**12, 1, 0)
    y1_true = np.where(y > 10 ** 11, 1, 0)
    y_act = np.where(y > 3.5 * (10 ** 10), 1, 0)
    y2_true = np.where(y > 10 ** 10, 1, 0)
    print(np.sum(y_true))
    print(np.sum(y1_true))
    print(np.sum(y_act))
    print(np.sum(y2_true))
    # exit()
    #
    #
    #
    # plt.plot(y)
    # plt.show()
    # exit()
    # Split the data


    X_train_temp, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_rate), random_state=seed)
    X_val_temp, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_rate, random_state=seed)

    X_train = X_train_temp[input_features]
    X_val = X_val_temp[input_features]

    if not os.path.exists("test_data"):
        os.makedirs("test_data")

    # save test data in model directory
    test = pd.ExcelWriter(f"{model_directory}/{output_features_tag}_model/test_data.xlsx")
    test_data = pd.concat([X_test, y_test], axis=1)  # Combine features and target variable
    test_data.to_excel(test, index=False)
    test.close()

    # save input features in model directory
    input_features_panda = pd.DataFrame(input_features, columns=["Input Features"])
    input_features_panda.to_csv(f"{model_directory}/{output_features_tag}_model/{num_neurons}_neurons_{num_layers}_layers/{approach}/input_features.txt", sep=" ")

    # save model parameters
    model_parameter_values = [approach, mode, train_rate, test_rate, num_neurons, num_layers, activ_func, learning_rate,
                        train_batch_size, val_batch_size, patience, training_epochs, seed, best_training_epoch]
    model_parameter_tags = ["approach", "mode", "train_rate", "test_rate", "neurons", "layers",
                            "activ_func", "learning_rate", "train_batch_size", "val_batch_size", "patience",
                            "training_epochs", "seed", "best_training_epoch"]
    model_parameters = pd.DataFrame([model_parameter_values], columns=model_parameter_tags)
    model_parameters.to_csv(f"{model_directory}/{output_features_tag}_model/{num_neurons}_neurons_{num_layers}_layers/{approach}/model_parameters.txt", sep=" ")


    if approach == "PyTorch":
        # Scale the data (fit only on training data)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # save scaler
        joblib.dump(scaler, f'{model_directory}/{output_features_tag}_model/{num_neurons}_neurons_{num_layers}_layers/{approach}/scaler.pkl')

        # Convert numpy arrays to float32
        X_train = X_train.astype('float32')
        y_train = y_train.values.astype('float32')
        X_val = X_val.astype('float32')
        y_val = y_val.values.astype('float32')

        # Set activation function

        if activ_func == "ReLU":
            activ_func = nn.ReLU()
        else:
            print("Error activation function.")

        # Initialize the MLP and set to float32
        tik = time.time()
        torch.manual_seed(seed)
        if mode == "Old":
            mlp = assist.MLP_old(np.shape([input_features])[-1], np.shape([output_features])[-1], num_neurons)
        elif mode == "New":
            mlp = assist.MLP(np.shape([input_features])[-1], np.shape([output_features])[-1], num_neurons, num_layers,
                             activ_func)
        else:
            print("Error mode.")
            exit()

        mlp = mlp.float()

        # Loss function and optimizer
        loss_function = assist.RMSELoss()
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=learning_rate) #, weight_decay=0.8)

        # Data loaders
        train_dataset = assist.Data_prep(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

        val_dataset = assist.Data_prep(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

        # Lists to store average relative losses per epoch
        rmse_train = []
        rmse_val = []
        mape_train = []
        mape_val = []

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0

        # keeping best model variables
        best_val_MAPE = float('inf')
        best_train_MAPE = float('inf')

        for epoch in range(training_epochs):
            total_train_loss = 0.0
            total_val_loss = 0.0
            total_mape_train = 0.0
            total_mape_val = 0.0
            num_train_data_points = 0
            num_val_data_points = 0
            # Training phase
            mlp.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.float(), targets.float()
                optimizer.zero_grad()
                outputs = mlp(inputs).squeeze()
                loss = loss_function(outputs, targets)
                mape_loss = mean_absolute_percentage_error(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * inputs.size(0)
                total_mape_train += mape_loss * inputs.size(0)
                num_train_data_points += inputs.size(0)

            # Validation phase
            mlp.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.float(), targets.float()
                    outputs = mlp(inputs).squeeze()
                    loss = loss_function(outputs, targets)
                    mape_loss = mean_absolute_percentage_error(outputs.detach().cpu().numpy(),
                                                               targets.detach().cpu().numpy())
                    total_val_loss += loss.item() * inputs.size(0)
                    total_mape_val += mape_loss * inputs.size(0)
                    num_val_data_points += inputs.size(0)

            # Calculate average losses for this epoch
            average_rmse_train_loss = 100 * total_train_loss / num_train_data_points
            average_rmse_val_loss = 100 * total_val_loss / num_val_data_points
            average_mape_train = 100 * total_mape_train / num_train_data_points
            average_mape_val = 100 * total_mape_val / num_val_data_points
            rmse_train.append(average_rmse_train_loss)
            rmse_val.append(average_rmse_val_loss)
            mape_train.append(average_mape_train)
            mape_val.append(average_mape_val)

            # Early stopping logic
            if average_rmse_val_loss < best_val_loss:
                best_val_loss = average_rmse_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

            # Save best model
            if average_mape_val < best_val_MAPE:
                best_val_MAPE = average_mape_val
                best_train_MAPE = average_mape_train
                best_epoch = epoch
                best_model = copy.deepcopy(mlp.state_dict())
                best_optim = copy.deepcopy(optimizer.state_dict())

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(
                    f'Epoch {epoch + 1}, Training Loss: {average_mape_train:.4f}%, '
                    f'Validation Loss: {average_mape_val:.4f}% | Current Best Model found at Epoch {best_epoch}, '
                    f'Training Loss: {best_train_MAPE:.4f}%, Validation Loss: {best_val_MAPE:.4f}%')

            if epoch == best_training_epoch:
                training_model_path = f'{model_directory}/{output_features_tag}_model/{num_neurons}_neurons_{num_layers}_layers/{approach}/mlp_training_model_epoch_{best_training_epoch}.pth'
                torch.save(mlp.state_dict(), training_model_path)
        print(time.time() - tik)

        # Retrieve best model
        mlp.load_state_dict(best_model)
        optimizer.load_state_dict(best_optim)

        # Save best model
        best_model_path = f'{model_directory}/{output_features_tag}_model/{num_neurons}_neurons_{num_layers}_layers/{approach}/best_mlp_model.pth'
        torch.save(best_model, best_model_path)

        # Print information about the best model (Validation)
        print(
            f'Ultimate Best Model found at epoch {best_epoch}, with Train MAPE: {best_train_MAPE:.4f}%, Validation MAPE: {best_val_MAPE:.4f}%')
        # Plot Average Relative Training Loss vs Validation Loss
        plt.figure()
        plt.plot(range(1, len(mape_train) + 1), mape_train, label='Relative Training Loss', color='k')
        plt.plot(range(1, len(mape_val) + 1), mape_val, label='Relative Validation Loss', color='r', linestyle='-.')
        plt.xlabel('Epochs')
        plt.ylabel('MAPE Loss [%]')
        plt.legend(('Training Loss', 'Validation Loss'))
        plt.title('MAPE Loss per Epoch')
        plt.legend()
        plt.grid()
        plt.show()

    elif approach == "TensorFlow":  # function for applying TensorFlow Keras for ANN
        tik = time.time()  # Initialize time
        print(tik)
        keras.utils.set_random_seed(seed)  # set seed
        normalizer = tf.keras.layers.Normalization(axis=-1)  # initiate normalizer layer for value normalization
        normalizer.adapt(X_train)  # train normalizer for training value data
        X_train_normal = normalizer(X_train).numpy()
        checkpoint_filepath = f'{model_directory}/{output_features_tag}_model/{num_neurons}_neurons_{num_layers}_layers/{approach}/best_mlp_model'
        model_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss',
                                                         mode='min', save_best_only=True)

        def build_and_compile_model(norm, num_neurons, num_layers, activation, regularization, seed):

            # Use HeNormal initializer with a seed for better weight scaling
            initializer = HeNormal(seed=seed)

            model = keras.Sequential()  # set a sequential ANN structure
            model.add(norm)  # Add normalization layer first

            # Add hidden layers
            for _ in range(num_layers):
                model.add(layers.Dense(num_neurons,
                                       activation=activation,
                                       kernel_regularizer=regularizers.L2(l2=regularization),
                                       kernel_initializer=initializer))
                if dropout > 0.0:
                    model.add(layers.Dropout(dropout))
            # Output layer
            model.add(layers.Dense(1, kernel_initializer=initializer))

            # Compile the model
            model.compile(loss=loss_func, metrics=['mape'],
                          optimizer=keras.optimizers.Adam(learning_rate))

            return model

        # Example Usage
        model = build_and_compile_model(norm=normalizer, num_neurons=num_neurons, num_layers=num_layers,
                                        activation='relu', regularization=reg, seed=seed)
        model.summary()

        # execute model
        model_run = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=train_batch_size,
                              verbose=show_progress, epochs=training_epochs, callbacks=model_callback)
        print(time.time() - tik)
        plt.rcParams.update({"font.size": 35})

        predict_model = keras.models.load_model(
            f"{model_directory}/{output_features_tag}_model/{num_neurons}_neurons_{num_layers}_layers/{approach}/best_mlp_model")

        train_loss, train_acc = predict_model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = predict_model.evaluate(X_val, y_val, verbose=0)

        print(f"Best model performance:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        def plot_loss(history):  # create function for plotting convergence process for training and validation losses
            plt.plot(history.history['loss'], 'k', label='Loss')
            plt.plot(history.history['val_loss'], 'k-.', label='Validation Loss')
            # plt.ylim([0, 10])
            plt.xlabel('Epoch [-]')
            plt.ylabel('MAPE [%]')
            # plt.title('ANN Training Progress')
            plt.legend()
            plt.grid(True)
            plt.show()

        # plot data
        plot_loss(model_run)

    else:
        print("Error ANN approach!")
        exit()