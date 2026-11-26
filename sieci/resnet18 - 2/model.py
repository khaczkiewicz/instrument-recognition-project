
import torch
import torch.nn as nn
from torchvision import models

class Resnet18(nn.Module):

    def __init__(self, dropout=0.3, weights_type="default", weights_path=None):
        super().__init__()

        if weights_type == "default":
            self.model = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )
        else:
            self.model = models.resnet18(weights=None)

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

        if weights_type == "custom":
            checkpoint = torch.load(weights_path, map_location="cpu")

            checkpoint = {
                key.replace("model.", "", 1): value
                for key, value in checkpoint.items()
            }

            self.model.load_state_dict(checkpoint)

    def forward(self, x):
        return self.model(x)

class Resnet18Variant(nn.Module):

    def __init__(
        self,
        dropout=0.3,
        weights_type="default",
        weights_path=None,
        kernel_size=7,
        stride=2,
        padding=3,
        maxpool=True
    ):
        super().__init__()

        if weights_type == "default":
            self.model = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )
        else:
            self.model = models.resnet18(weights=None)

        # Zmiana pierwszej warstwy na wejście jednokanałowe
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        # Opcjonalne wyłączenie maxpool
        if not maxpool:
            self.model.maxpool = nn.Identity()

        # Zmiana warstwy wyjściowej na 7 instrumentów
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 7)
        )

        if weights_type == "custom":
            checkpoint = torch.load(weights_path, map_location="cpu")

            checkpoint = {
                key.replace("model.", "", 1): value
                for key, value in checkpoint.items()
            }

            self.model.load_state_dict(checkpoint)

    def forward(self, x):
        return self.model(x)
