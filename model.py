from torchvision import models
import torch
import torch.nn as nn

class gesture_cls(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet101 = models.resnet101(pretrained=True)
        self.classifier =nn.Sequential(nn.Linear(1000,500),
                                       nn.BatchNorm1d(500),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(500,100),
                                       nn.BatchNorm1d(100),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(100,10),
                                       nn.BatchNorm1d(10)
        )

    def forward(self, x):
        x = self.resnet101(x)
        x = self.classifier(x)

        return x
