import torch
from torch import nn

class HelloWorldModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input + 1

HelloWorldModule = HelloWorldModule()
input_data = torch.tensor(1)
output_data = HelloWorldModule(input_data)
print(output_data)
