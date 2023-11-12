import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


class TorchNN_train(nn.Module):

  # Initializes the weights and biases
  def __init__(self) -> None:
    super().__init__()
    self.w00 = nn.Parameter(torch.tensor(1.7) , requires_grad = False)
    self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
    self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

    self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
    self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
    self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

    self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)
  
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

  def train(self, inputs, labels, total_loss):
    optimizer = SGD(model.parameters(), lr=.1)
    print("Starting bias", model.final_bias)
    for epoch in range(100):
      for iteration in range(len(inputs)):
        input_i = inputs[iteration]
        label_i = labels[iteration]
        output_i = model(input_i)
        loss = (output_i - label_i)**2
        loss.backward()
        total_loss += float(loss)
      print(total_loss)
      if(total_loss < 0.0001):
        print("Num of steps" , str(epoch))
        break
      optimizer.step()
      optimizer.zero_grad()
    return total_loss


inputs = torch.tensor([0.0, .5, 1.0])
labels = torch.tensor([0,1,0])
model = TorchNN_train()
print(model.parameters())
print("Bias before optimization: " , str(model.final_bias.data))

arbitarary_total_loss = 0.0

model.train(inputs, labels, arbitarary_total_loss)
print("Total loss after training: ", model.final_bias)

input = torch.linspace(start=0, end=1, steps=11)
print(input)
output = model(input)
print(output)
# sns.set(style='whitegrid')
# sns.lineplot(x=input,
#              y=output.detach(),
#              color='green',
#              linewidth=2.5)



    
