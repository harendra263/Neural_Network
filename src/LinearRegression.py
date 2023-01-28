import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
Linear Regression:
    .   Allows us to understood the relationship between continuous variables
    .   Example
            x: independent variable
                . weight
            y: dependent variable
                . height
    
    .   y = ax + b

Aim of Linear Regression
    .   Minimize the distance between the points and the line (y = αx + β)
    .   Adjusting
        .   Coefficient: α
        .   Bias/intercept: β
"""

x_values = list(range(11))
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
print(x_train.shape)

y_values = [2.25 * i for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
print(y_train)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    

    def forward(self, x):
        return self.linear(x)
    
input_dim = 1
output_dim = 1

model = LinearRegression(input_dim, output_dim)

# -----------------
# USE GPU AS MODEL
# -----------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device=device)

# Instantiate Loss Class
criterion = nn.MSELoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


epochs= 100

for epoch in range(epochs):
    epoch += 1
    
    # Convert the numpy array to torch tensor
    inputs = torch.from_numpy(x_train).requires_grad_().to(device)
    labels = torch.from_numpy(y_train).to(device)

    # Clear gradients w.r.t parameters
    optimizer.zero_grad()

    # Forward to get output
    outputs = model(inputs)

    # Calculate loss
    loss = criterion(outputs, labels)

    # getting gradients w.r.t parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    print(f'epoch {epoch}, loss {loss.item()}')


# Looking at the predicted values
predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
print(predicted)
print(y_train)

# Clear figure
plt.clf()

# plot true data
plt.plot(x_train, y_train, 'go', label='True Data', alpha=0.5)

plt.plot(x_train, predicted, '--', label='Predictions', alpha =0.5)

plt.legend(loc='best')
plt.show()

