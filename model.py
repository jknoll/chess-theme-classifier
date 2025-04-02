# Network derived from: https://github.com/pytorch/examples/blob/main/mnist/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, num_labels=62):  # Default to original size for backward compatibility
        super(Model, self).__init__()
        # 1 input image channel, 32 output channels, 3x3 convolution
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # Calculate the size after convolutions and pooling
        # After conv1: (8-2)x(8-2) = 6x6x32
        # After pool1: 3x3x32
        # After conv2: 1x1x64
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 512)  # Increased intermediate layer size
        self.fc2 = nn.Linear(512, 256)  # Added another layer
        self.fc3 = nn.Linear(256, num_labels)  # Output layer
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, input, debug=False):
        # First convolution and pooling
        x = F.relu(self.conv1(input))
        if debug: print(f"After conv1: {x.shape}")
        
        x = F.max_pool2d(x, 2)
        if debug: print(f"After pool1: {x.shape}")
        
        # Second convolution and pooling
        x = F.relu(self.conv2(x))
        if debug: print(f"After conv2: {x.shape}")
        
        # Flatten
        x = torch.flatten(x, 1)
        if debug: print(f"After flatten: {x.shape}")
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        if debug: print(f"After fc1: {x.shape}")
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        if debug: print(f"After fc2: {x.shape}")
        
        x = self.fc3(x)
        if debug: print(f"After fc3: {x.shape}")
        
        return x