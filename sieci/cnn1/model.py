# Biblioteki
import torch
import torch.nn as nn
import torch.nn.functional as F

class InstrumentCNN(nn.Module):
    """
    Architektura opiera się na jednej sieci CNN przetwarzającej
    cały mel-spektrogram. Na wyjściu sieć zwraca wektor
    7 wartości, z których każda odpowiada jednemu instrumentowi.
    """
    def __init__(self, dropout2, dropout3):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.conv3 = nn.Conv2d(64,128,3,1)
        self.dropout2 = nn.Dropout2d(dropout2)
        self.dropout3 = nn.Dropout2d(dropout3)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc3 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc3(x)

        return x