#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:29:14 2022
Resnet 18 and 34
@author: ashrith
"""

import torch
import torchvision
import torch.nn as nn

# all variables with self prefix declared in __init__() are public to class and the rest are function specific


class block(nn.Module):  # to run each block
    def __init__(
        self, input_channels, output_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class Resnet(
    nn.Module
):  # to build and forward the initial convs and each of the layers
    def __init__(self, block, layer_multiple, input_channel, output_features):
        super(Resnet, self).__init__()
        self.block_out = 64
        self.intermediate = 64
        self.conv1 = nn.Conv2d(
            input_channel, self.block_out, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nn.BatchNorm2d(self.block_out)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(
            block, self.block_out * 1, layer_multiple[0], stride=1
        )
        self.layer2 = self.make_layer(
            block, self.block_out * 2, layer_multiple[1], stride=2
        )
        self.layer3 = self.make_layer(
            block, self.block_out * 4, layer_multiple[2], stride=2
        )
        self.layer4 = self.make_layer(
            block, self.block_out * 8, layer_multiple[3], stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.block_out * 8, output_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layer(self, block, out_ch, layer_multiple, stride, padding=1):
        identity_downsample = None
        layers = []
        if self.intermediate != out_ch:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.intermediate, out_ch, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_ch),
            )
            layers.append(
                block(self.intermediate, out_ch, identity_downsample, stride=stride)
            )
        self.intermediate = out_ch
        if self.intermediate == 64:
            num_iterations = layer_multiple
        else:
            num_iterations = layer_multiple - 1
        for i in range(num_iterations):
            layers.append(block(self.intermediate, out_ch))
        return nn.Sequential(*layers)


def ResNet18(image_channels=3, output_channels=10):
    return Resnet(block, [2, 2, 2, 2], image_channels, output_channels)


def ResNet50(image_channels=3, output_channels=10):
    return Resnet(block, [3, 4, 6, 3], image_channels, output_channels)


if __name__ == "__main__":
    model = ResNet18(3, 10)
    y = model(torch.randn(4, 3, 224, 224)).to("cpu")
    print("Final size", y.size())
