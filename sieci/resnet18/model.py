import torch
import torch.nn as nn
from torchvision import models


class Resnet(nn.Module):
    """
    Pojedyncza sieć ResNet18 rozpoznająca jednocześnie
    obecność 7 instrumentów na podstawie log-mel spektrogramu.

    Wejście:
        (batch_size, 1, 80, czas)

    Wyjście:
        (batch_size, 7)
    """

    def __init__(self, dropout=0.3):
        super().__init__()

        # ResNet bez wag pretrenowanych
        #self.model = models.resnet18(weights=None)

        # ResNet18 z wagami pretrenowanymi na ImageNet
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Zmiana pierwszej warstwy na wejście jednokanałowe
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Zmiana warstwy wyjściowej na 7 instrumentów
        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 7)
        )

    def forward(self, x):
        return self.model(x)
