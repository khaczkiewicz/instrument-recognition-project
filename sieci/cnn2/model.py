import torch
import torch.nn as nn

class InstrumentCNN(nn.Module):

    def __init__(
        self,
        dropout2=0.1,
        dropout3=0.3,
        weights_type="default",
        weights_path=None
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)

        self.pool = nn.MaxPool2d(2)

        self.dropout2 = nn.Dropout2d(dropout2)
        self.dropout3 = nn.Dropout2d(dropout3)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.output = nn.Linear(128, 1)

        if weights_type == "custom":
            checkpoint = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(checkpoint)

    def forward(self, x):
        x = torch.relu(self.conv1(x))

        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout3(x)

        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        return self.output(x)

class MultiInstrumentCNN(nn.Module):

    def __init__(
        self,
        num_instruments,
        dropout2=0.1,
        dropout3=0.3,
        weights_type="default",
        weights_path=None
    ):
        super().__init__()

        self.models = nn.ModuleList([
            InstrumentCNN(
                dropout2,
                dropout3,
                weights_type,
                weights_path
            )
            for _ in range(num_instruments)
        ])

    def forward(self, x):
        outputs = [model(x) for model in self.models]

        return torch.cat(outputs, dim=1)