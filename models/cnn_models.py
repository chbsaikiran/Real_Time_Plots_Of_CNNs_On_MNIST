import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, conv1_layers, conv2_layers, conv3_layers):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv1_layers, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_layers, conv2_layers, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv2_layers, conv3_layers, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc = nn.Linear(conv3_layers * 3 * 3, 10)  # Direct mapping to 10 classes
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)  # Apply log_softmax for numerical stability

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features