import torch
import torch.nn as nn
from torchvision import models

class ByteSightResNet(nn.Module):
    """
    Modular ResNet model for malware image classification.
    Automatically handles 1-channel (grayscale) input.
    """
    def __init__(self, num_classes=9, backbone='resnet18', pretrained=True):
        super(ByteSightResNet, self).__init__()
        
        # Load backbone
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Backbone {backbone} not supported.")
            
        # 1. Modify the first layer to accept 1-channel (grayscale) instead of 3 (RGB)
        # We copy the weights from the first channel of the pretrained model if available
        old_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(1, old_conv.out_channels, 
                                    kernel_size=old_conv.kernel_size, 
                                    stride=old_conv.stride, 
                                    padding=old_conv.padding, 
                                    bias=old_conv.bias)
        
        if pretrained:
            with torch.no_grad():
                # Average the weights across RGB channels or just take the first one
                self.model.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        # 2. Modify the final layer (Fully Connected) to match our number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)

def get_model(num_classes, backbone='resnet18', pretrained=True):
    return ByteSightResNet(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
