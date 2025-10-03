import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class Model(nn.Module):
    """
    
    """

    def __init__(self, in_features: int = 4, h1: int = 8, h2: int = 9, out_features: int = 3):

        super().__init__() # Intiantiate the nn module

        # Define the structure: In -> Layer 1 -> Layer 2 -> Out using Fully Connected layers (FC)
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        
    def forward(self, x):

        x = F.relu(self.fc1(x)) # relu: Rectified Linear Unit (outputs the input if positive, else outputs zero)
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
    
# Pick manual seed for randomization
torch.manual_seed(41)

# Instance
model = Model()

# Load data
url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)

my_df["species"] = my_df["species"].replace("setosa", 0.0)
my_df["species"] = my_df["species"].replace("versicolor", 1.0)
my_df["species"] = my_df["species"].replace("virginica", 2.0)
print(my_df)

# Train, test and split
x = my_df.drop("species", axis = 1)
y = my_df["species"]

X = x.values
y = y.values

X_train, X_test, y_train, y_test  = train_test_split(
    X, y, test_size = 0.2, random_state = 41
)

# Convert features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert reponse to integers tensors
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Criterion to measure model error
criterion = nn.CrossEntropyLoss()
# Choose optimiser and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

#print(model.parameters)
# Train model
# Epochs -> number of runs of the training data through the nn
epochs = 1000
losses = []
for i in range(epochs):
    
    # Forward with a prediction
    y_pred = model.forward(X_train)

    # Measure loss
    loss = criterion(y_pred, y_train)

    # Keep track of losses 
    losses.append(loss.detach().numpy())

    # Print results every 10 epochs
    if i % 10 == 0:
        print(f"Epoch: {i}, Loss: {loss}")

    # Backward propagation: Feed the results back to the network to 
    # optimize the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# plot results
plt.plot(range(epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
#plt.show()

with torch.no_grad(): # Turns off backpropagation - Sends results back to the start
    y_eval = model.forward(X_test)  # Sends 
    loss = criterion(y_eval, y_test)

print(loss)

# Predict based on X_test
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        # Type of flower our network thinks the flower is
        print(f'{i+1}.) {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

        # Find out if resutls are correct or not
        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f"We got {correct} correct!")

# Save the neural network
torch.save(model.state_dict(), "test_model.pt")
# Load saved model
new_model = Model()
new_model.load_state_dict(torch.load('test_model.pt'))
print(new_model.eval())