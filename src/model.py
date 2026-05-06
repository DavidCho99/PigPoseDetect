import torch
import torch.nn as nn
from torchvision.transforms import v2
import random

from typing import Any

'''
from answers of Dr.Monico.
A funnel structure that doubles the number of filters, such as 32 -> 64 -> 128 Also
    Source: The 2014 paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" by researchers at the University of Oxford (Simonyan & Zisserman).
    This is the design approach proposed by the VGGNet architecture presented in this paper.
    Doubling the number of channels (filters) during the pooling process has now become the textbook standard for all CNN designs.

The principle that early layers learn simple features (lines), while deeper layers learn complex features (shapes)
    Source: The 2013 research paper "Visualizing and Understanding Convolutional Networks" (Zeiler & Fergus) and Chapter 9 (Convolutional Networks) of the book "Deep Learning" (2016) by Ian Goodfellow,
      one of the pioneers of deep learning. These materials mathematically and experimentally proved what features each layer learns by visualizing the inside of the neural network.

The guideline of using 3 to 4 layers and Dropout for a small dataset of images
    Source: Chapter 5 (Deep Learning for Computer Vision) of the book "Deep Learning with Python"(2017) by François Chollet, the creator of the Keras library.
    This adopts the methodology for building a baseline model to prevent overfitting when training a CNN from scratch on a small dataset.

Using 128 to 512 neurons in the Fully Connected (Dense) layer and assigning powers of 2
    Source: Professor Andrew Ng's Coursera course "Deep Learning Specialization" at Stanford University, alongside standard hardware architecture optimization practices in computer science.
    This is based on the computer architecture fact that setting the number of neurons or batch size to a power of 2—such as 64, 128, 256, or 512—is the most efficient for computation speed due to
    the memory block allocation structures of GPUs/CPUs.


Most of my project structure follow this -> https://careers.saigontechnology.com/blog-detail/build-deep-learning-model-for-the-image-classification-task-part1-pytorch-model
Information from here I couldn't attend the class when class cover this https://learnopencv.com/understanding-convolutional-neural-networks-cnn/
module implement from here-> https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
'''
cfg = {"A": [32, "M", 64, "M", 128, "M", 256, "M", 512, "M"]}


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # Conv -> BN -> ReLU
            '''
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            '''

            # padding is required if not image will be cutting out and model will get image with no useful info.
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# I copy the most of the code from https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py and honestly some of the code was hard to understand.
# But if I delete the code it doesn't work, so I added.


class PigPosture_CNN(nn.Module):
    # One hidden layer with 128 nodes, densely connected.
    def __init__(self, num_classes=5, init_weights: bool = True, dropout: float = 0.5):
        super().__init__()
        # This is my feature capture network format.
        # [32, "M", 64, "M", 128, "M", 256, "M", 512, "M"]
        self.features = make_layers(cfg['A'])
        # What I understand, it's from the VGG.
        # I guess it's 7 beacuse it's prime number. we can't divide it by 2
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # VGG originally use 4096 for last hidden layer but I don't want to do it. I thought it was computationally too big.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            # Dropout uses y = mask * x / (1 - p), where mask ~ Bernoulli(1 - p).
            # I chose p=0.5, so about half of the activations are dropped during training.
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

        self.loss_func = nn.functional.cross_entropy
        self.optimizer = torch.optim.Adam(self.parameters())

        self.to(device)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_epoch(self, train_loader):
        """ Train on the entire dataset once (one epoch).
            Return value: (loss, accuracy)."""
        self.train()
        total_loss = 0
        correct = 0
        AugTransform = v2.RandomAffine(
            degrees=20, translate=(0.12, 0.12), shear=(-15, 15))
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            # Perform the augmentation transform with probability 0.75.
            if random.random() < 0.75:
                data = AugTransform(data)
            # Evaluate the model and loss.
            output = self(data)
            loss = self.loss_func(output, target)
            # Gather some intermediate statistics
            total_loss += loss.item()*len(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum()
            # Take a gradient descent step.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        N = len(train_loader.dataset)
        return total_loss/N, correct.item()/N  # item becuae it's on the GPU

    def assess(self, loader):
        """Evaluate the model on the given set.
           Return value: (loss, accuracy)."""
        self.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in loader:
                data = data.to(device)
                target = target.to(device)

                output = self(data)
                loss = self.loss_func(output, target)
                total_loss += loss.item()*len(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        N = len(loader.dataset)
        return total_loss/N, correct/N
