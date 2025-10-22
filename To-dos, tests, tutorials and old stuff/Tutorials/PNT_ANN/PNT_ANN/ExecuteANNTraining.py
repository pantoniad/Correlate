import ANNTraining as frf
#

approach = "TensorFlow"  # Set mode : Approach 1 sets dataset as whole and randomly splits it
#                                 Approach 2 sets dataset according to Design_id
#                                 *** For more information refer to D3.1, section 3.3 ***
mode = "New"  # Define Prediction model mode : "Old" for old versioning of MLP definition | Models before 18/4/2024
# --> Requires definition of layers and activation function in AssistFunctions, MLP_old class
#                                              "New" for new versioning of MLP definition | Models after 18/4/2024
# ***** BEWARE : Even if a model has identical input_layer/hidden_layer/output_layer structure, the versioning of MLP
# definition must be identical to that for which the model was trained, otherwise errors occur : BEWARE *****
DataSetDirectory = "flightpath_parallel_100K"  # Set Directory  *** The model selects the dataset
# by itself according to the output feature ***
train_rate = 0.8  # Ratio of training to total datapoints [-]
test_rate = 0.5  # Ratio of testing to (testing + validation) datapoints [-]
model_directory = "v2"  # Define directory *** ONLY IF NEEDED *** for model output
# !!!! BEWARE !!!! a model directory IS created within the DataSetDirectory, this is only for having multiple models in
# the same directory and for the same output feature
neurons = 512  # Define number of neurons in each hidden layer [-]
layers = 8  # Define number of hidden layers [-]
activ_func = "ReLU"  # Define the activation function [-] *** as of 23/4/2024 only "ReLU" ***
learning_rate = 1e-3  # Define ANN learning rate [-]
reg = 5*1e-5  # Define the l2 regularization rate [-]
dropout = 0  # Define the dropout rate [-]
loss_func = "mape"  # define the training loss function for ANN optimization [-]
train_batch_size = 100  # Define ANN training batch size [-]
val_batch_size = 100  # Define ANN validation batch size [-]
patience = 100  # Define the number of epochs that, after no improvement to RMSE, early stopping will commence. [-]
training_epochs = 3000  # Define the maximum number of training epochs [-]

input_features = ['Distance [m]','Cruise Velocity [m/s]', 'Cruise Altitude [m]', 'Ascent 2 Velocity',
                'Ascent 2 Rate', 'Ascent Altitude', 'Ascent 1 Velocity', 'Descent 1 Velocity', 'Ascent 1 Rate',
                'Descent 1 Rate', 'Loiter Velocity', 'Descent 2 Velocity']  # input feature order for 80K / datahoarder

# input_features = ['Distance [m]','Cruise Velocity [m/s]', 'Cruise Altitude [m]', 'Ascent 2 Velocity',
                # 'Ascent 2 Rate', 'Ascent Altitude', 'Ascent 1 Velocity',  'Ascent 1 Rate', 'Descent 1 Velocity',
                # 'Descent 1 Rate', 'Loiter Velocity', 'Descent 2 Velocity']  # input feature order for 100K / datahoarder_v2

output_features = "Energy_Consumption"  # Either Power[W] or Power_Rate for power |
#                                      TSFC[g/kN/s] or Fuel_Flow[kg/s] for TSFC
seed = 43  # Set Random Seed
best_training_epoch = 200  # Set the epoch for keeping a random model
show_progress = 2  # option to display progress when training in TensorFlow Keras: 0 for no progress
#                                                                                  1 for progress bar
#                                                                                  2 for line status
# *** BEWARE *** MUST NOT BE GREATER THAN training_epochs


frf.train(approach, mode, DataSetDirectory, train_rate, test_rate, model_directory, neurons, layers,
          activ_func, learning_rate, reg, dropout, loss_func, train_batch_size, val_batch_size, patience,
          training_epochs, seed, input_features, output_features, best_training_epoch, show_progress)
