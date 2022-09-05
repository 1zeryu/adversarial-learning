from email.mime import image
from random import shuffle
from sys import stderr
from visdom import Visdom
from Dataset.dataset import data
from adversarial.Gaussian import GaussianNoise
from torchvision import datasets
import torchvision
import torch
import matplotlib.pyplot as plt

trainloader, testloader, num_class, trainset, testset = data()


class_idx = testset.classes

examples = iter(testloader)
inputs, targets = next(examples)

atk = GaussianNoise(sigma=0.5)
adv_images = atk(inputs, targets)

vis = Visdom(env='main')
    
vis.image(torchvision.utils.make_grid(inputs[0]))
vis.image(torchvision.utils.make_grid(adv_images[0]))