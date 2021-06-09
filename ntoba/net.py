from torch import nn
import torch


class Net(nn.Module):

    def __init__(self, input_features, hidden_features, output_features):

        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, output_features)


    def forward(self, x):

        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x
