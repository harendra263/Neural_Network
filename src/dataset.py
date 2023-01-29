import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

print(len(train_dataset))

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

print(test_dataset)
