import torch
import torch.nn as nn
from torchvision import models

class TrueLensFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.spatial = models.resnet18(pretrained=True)
        self.spatial.fc = nn.Identity()

        self.freq = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(512 + 32, 2)

    def forward(self, img, fft):
        spatial_feat = self.spatial(img)
        freq_feat = self.freq(fft).view(fft.size(0), -1)
        fused = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.classifier(fused)
