# Network derived from: https://github.com/pytorch/examples/blob/main/mnist/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(54, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward(self, input, debug=False):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = F.relu(self.conv1(input))
        if (debug): 
            print(f"c1.shape: {c1.shape}")

        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 6, 14, 14) Tensor
        s2 = F.max_pool2d(c1, (2, 2))
        if (debug): 
            print(f"s2.shape: {s2.shape}")
     
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s2 = torch.flatten(s2, 1)
        if (debug): 
            print(f"s2.shape: {s2.shape}")

        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s2))
        if (debug): 
            print(f"f5.shape: {f5.shape}")

        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it
        #  uses RELU activation function
        f6 = F.relu(self.fc2(f5))
        if (debug): 
            print(f"f6.shape: {f6.shape}")

        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = self.fc3(f6)
        if (debug): 
            print(f"output.shape: {output.shape}")

        # Apply sigmoid activation to constrain output between 0 and 1
        output = torch.sigmoid(output)
        if (debug): 
            print(f"output after sigmoid.shape: {output.shape}")

        return output