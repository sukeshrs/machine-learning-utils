import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


class TorchNN(nn.Module):

  # Initializes the weights and biases
  def __init__(self) -> None:
    super().__init__()
    self.w00 = nn.Parameter(torch.tensor(1.7) , requires_grad = False)
    self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
    self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

    self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
    self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
    self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

    self.final_bias = nn.Parameter(torch.tensor(-16.0), requires_grad=False)
  
  # Does a forward pass and calculates the output value for the input
  def forward(self, input):
    top_input = input * self.w00 + self.b00
    top_relu_output = F.relu(top_input)
    scaled_top_relu_output = top_relu_output * self.w01
    bottom_input = input * self.w10 + self.b10
    bottom_relu_output = F.relu(bottom_input)
    scaled_bottom_relu_output = bottom_relu_output * self.w11

    input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

    output = F.relu(input_to_final_relu)

    return output


input = torch.linspace(start=0, end=1, steps=11)
print(input)
model = TorchNN()
output = model(input)
print(output)

sns.set(style='whitegrid')
sns.lineplot(x=input,
             y=output,
             color='green',
             linewidth=2.5)

plt.ylabel = 'Effectiveness'
plt.xlabel = 'dose'


    
