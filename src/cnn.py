import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from dataset import train_dataset, test_dataset


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper parameters
n_iters = 3000
batch_size = 50
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)
learning_rate = 0.01

# dataset has PILImage image of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

# Building a Pytorch Convolutional Network

print("Train Dataset: ",train_dataset.train_data.size())
print("Labels: ", train_dataset.targets.size())
print("Test Dataset: ", test_dataset.test_data.size())
print("Test Labels: ", test_dataset.targets.size())

# MODEL -A

class CNNModel(nn.Module):
    def __init__(self) -> None:
        super(CNNModel, self).__init__()

        # Convolution-1
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # Max pooling
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution-2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        # Max pooling
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 
        self.fc1 = nn.Linear(32 * 7 * 7, 10)


    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool out
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32 * 7 * 7)
        out = out.view(out.size(0), -1)
        
        # Linear function (readout)
        out = self.fc1(out)
        return out







