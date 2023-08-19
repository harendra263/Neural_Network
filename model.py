import torch
import torch.nn.functional as F
import torch.nn as nn
import config

class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()

        self.layer1 = nn.Linear(input_size, 24)
        self.layer2 = nn.Linear(24, 24)
        self.layer3 = nn.Linear(24, output_size)

    
    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        return self.layer3(x2)
    

        