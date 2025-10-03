import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 =  nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, input):
        
        # Convolutional layer C1
        c1 = F.relu(self.conv1(input))
        
        # Subsampling Layer S2
        s2 = F.max_pool2d(c1, (2,2))

        # Convolution layer C3
        c3 = F.relu(self.conv2(s2))

        # Subsampling layer S4. No parameters, no outputs
        s4 = F.max_pool2d(c3, 2)
        s4 = torch.flatten(s4,1) # Flatten

        # Fully connected layer F5
        f5 = F.relu(self.fc1(s4))

        # Fully connected layer F6
        f6 = F.relu(self.fc2(f5))

        # Outuput
        output = self.fc3(f6)
        return output
    
net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1,10))

# Loss function
output = net(input)
target = torch.randn(10)
target = target.view(-1,1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


