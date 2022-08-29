from sys import stderr
from Dataset.dataset import data
from adversarial.Gaussian import GaussianNoise
trainloader, testloader, num_classes, trainset, testset = data()

examples = iter(trainloader)
inputs, targets = next(examples)

atk = GaussianNoise(sigma=0.1)
adv_images = atk(inputs)

from visdom import Visdom

vis = Visdom(env='main')

vis.images(inputs[:16], 8, win='images/original')
vis.images(adv_images[:16], 8, win='images/adversarial')


