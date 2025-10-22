import Decision_Trees as dt
#

approach = "Decision_Trees"  # Set mode : Decision_Trees sets basic Decision Trees algorithm
#                                          TensorFlow_ANN sets TensorFlow MLP for training
#                                 *** For more information refer to D3.1, section 3.3 ***
DataSetDirectory = "engage-hackathon-2025/year=2024/month=11/day=16/hour=19"  # Set Directory  *** The model selects the dataset
DataSetDoc = "54bc0046968d47baa08a1936a523a7f8.snappy.parquet"
# by itself according to the output feature ***
train_rate = 0.8  # Ratio of training to total datapoints [-]
model_directory = None  # Define directory *** ONLY IF NEEDED *** for model output
# !!!! BEWARE !!!! a model directory IS created within the DataSetDirectory, this is only for having multiple models in
# the same directory and for the same output feature
max_depth = 5  #
min_samples_split = 8  #
min_samples_leaf = 5 #
criterion = "squared_error"  # Classification https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#               Regression https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
task = "Regression"  # Regression or Classification

input_features = None
# Define input features for the ANN

output_features = None  # Define the output feature or features for the model
seed = 42  # Set Random Seed
best_training_epoch = 200  # Set the epoch for keeping a random model
show_progress = 2  # option to display progress when training in TensorFlow Keras: 0 for no progress
#                                                                                  1 for progress bar
#                                                                                  2 for line status
# *** BEWARE *** MUST NOT BE GREATER THAN training_epochs


dt.train(approach, DataSetDirectory, DataSetDoc, train_rate, model_directory, max_depth, min_samples_split, min_samples_leaf, criterion, task,
          seed, input_features, output_features)
