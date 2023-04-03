#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:37:21 2022

@author: ashrith
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# all variables with self prefix declared in __init__() are public to class and the rest are function specific
"""
Class Block:
"""


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


"""
Class Resnet:
"""


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

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyper-parameters
    num_epochs = 5
    batch_size = 4
    learning_rate = 0.001
    model = ResNet18(3, 10)

    # transform:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))

    # model:
    model = model.to(device)

    # Loss function and optimser
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    start_time = time.time()
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 2000 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}"
                )
    print("Finished Training in time", (time.time() - start_time) / 60, "mins")
    PATH = "./cnn.pth"
    torch.save(model.state_dict(), PATH)

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
        acc = 100.0 * n_correct / n_samples
        print(f"Accuracy of the network: {acc} %")
        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f"Accuracy of {classes[i]}: {acc} %")
