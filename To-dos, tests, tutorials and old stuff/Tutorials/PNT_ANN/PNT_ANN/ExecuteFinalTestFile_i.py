import FinalTestFile_i as ftf


DataSetFile = "flightpath_parallel_100K/v2/Energy_Consumption_model/test_data.xlsx"

# Set DataSet File  *** For Power & TSFC it should vary ***
sheet_name = "Sheet1"
model_directory = "Architecture_2/Dataset19_Arch2_241027/Fuel_Flow_model"  # Define characteristic tag for files / directories
neurons = 256  # Define number of neurons in each hidden layer [-]
layers = 8  # Define number of hidden layers [-]
activ_func = "ReLU"  # Define the activation function [-] *** as of 23/4/2024 only "ReLU"
mode = "New"  # Define Prediction model mode : "Old" for old versioning of MLP definition | Models before 18/4/2024
# --> Requires definition of layers and activation function in AssistFunctions, MLP_old class
#                                              "New" for new versioning of MLP definition | Models after 18/4/2024
# ***** BEWARE : Even if a model has identical input_layer/hidden_layer/output_layer structure, the versioning of MLP
# definition must be identical to that for which the model was trained, otherwise errors occur : BEWARE *****

input_features = ['Distance [m]','Cruise Velocity [m/s]', 'Cruise Altitude [m]', 'Ascent 2 Velocity',
                'Ascent 2 Rate', 'Ascent Altitude', 'Ascent 1 Velocity', 'Descent 1 Velocity', 'Ascent 1 Rate',
                'Descent 1 Rate', 'Loiter Velocity', 'Descent 2 Velocity']
#                   Architecture_2, Arhictecture_3, Architecture_4 default input featuers  *** UNCOMMENT IF USING ***
output_features = "Energy_Consumption"  # Either Power[W] or Power_Rate for power or TSFC[g/kN/s] or Fuel_Flow[kg/s] for
# TSFC
approach = "TensorFlow"



ftf.test(DataSetFile, sheet_name, model_directory, neurons, layers, activ_func, mode, input_features, output_features, approach)
