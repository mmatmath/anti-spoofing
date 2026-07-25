import torch
from torch import nn


class MaxFeatureMap(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_half, second_half = torch.chunk(
            x,
            chunks=2,
            dim=self.dim,
        )
        return torch.maximum(first_half, second_half)


class LCNN(nn.Module):
    def __init__(
        self,
        n_classes: int = 2,
        dropout: float = 0.75,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            MaxFeatureMap(dim=1),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            MaxFeatureMap(dim=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MaxFeatureMap(dim=1),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.BatchNorm2d(48),
            nn.Conv2d(
                in_channels=48,
                out_channels=96,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            MaxFeatureMap(dim=1),
            nn.BatchNorm2d(48),
            nn.Conv2d(
                in_channels=48,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MaxFeatureMap(dim=1),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            MaxFeatureMap(dim=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MaxFeatureMap(dim=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            MaxFeatureMap(dim=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            MaxFeatureMap(dim=1),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=32 * 53 * 37,
                out_features=160,
            ),
            MaxFeatureMap(dim=1),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(80),
        )
        self.classifier = nn.Linear(
            in_features=80,
            out_features=n_classes,
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight)

    def forward(
        self,
        data_object: torch.Tensor,
        **batch,
    ) -> dict[str, torch.Tensor]:
        features = self.net(data_object)
        embedding = self.embedding(features)
        logits = self.classifier(embedding)
        return {
            "logits": logits,
        }

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info