from random import shuffle
import torchvision 
from torchvision import transforms
import torch

dataset_dir = {
        'cifar10':'./Dataset/cifar10',
        'cifar100':'./Dataset/cifar100',
        'mnist':'./Dataset/mnist',
        'fashionmnist':'./Dataset/fashionmnist'
    }

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'mnist': (0.5, 0.5, 0.5)
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'mnist': (0.5, 0.5, 0.5),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

CIFAR10_transform_train1 = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean['cifar10'], std['cifar10']),
        ])

CIFAR10_transform_train2 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean['cifar10'], std['cifar10']),
        ]) # meanstd transformation

CIFAR10_transform_test1 = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean['cifar10'], std['cifar10']),
        ])

CIFAR10_transform_test2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean['cifar10'], std['cifar10']),
        ])

CIFAR100_transform_train1 = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean['cifar100'], std['cifar100']),
        ])

CIFAR100_transform_train2 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean['cifar100'], std['cifar100']),
        ]) # meanstd transformation

CIFAR100_transform_test1 = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean['cifar100'], std['cifar100']),
        ])

CIFAR100_transform_test2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean['cifar100'], std['cifar100']),
        ])

MNIST_transform_train1 = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean['mnist'], std['mnist']),
        ])

MNIST_transform_train2 = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean['mnist'], std['mnist']),
        ]) # meanstd transformation

MNIST_transform_test1 = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean['mnist'], std['mnist']),
        ])

MNIST_transform_test2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean['mnist'], std['mnist']),
        ])

num_classes = {'cifar10': 10,
               'cifar100': 100,
               'mnist': 10,}

def data(model='resnet18', dataset='cifar10', batch_size=128, num_workers=0):
    if model == 'pretrainedvit':
        if dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root=dataset_dir['cifar10'], train=True, download=True, transform=CIFAR10_transform_train1)
            testset = torchvision.datasets.CIFAR10(root=dataset_dir['cifar10'], train=False, download=False, transform=CIFAR10_transform_test1)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                                    shuffle=False, num_workers=num_workers)
        if dataset == 'cifar100':
            trainset = torchvision.datasets.CIFAR10(root=dataset_dir['cifar100'], train=True, download=True, transform=CIFAR100_transform_train1)
            testset = torchvision.datasets.CIFAR10(root=dataset_dir['cifar100'], train=False, download=False, transform=CIFAR100_transform_test1)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                                    shuffle=False, num_workers=num_workers)
        if dataset == 'mnist': 
            trainset = torchvision.datasets.CIFAR10(root=dataset_dir['mnist'], train=True, download=True, transform=MNIST_transform_train1)
            testset = torchvision.datasets.CIFAR10(root=dataset_dir['mnist'], train=False, download=False, transform=MNIST_transform_test1)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                                    shuffle=False, num_workers=num_workers)
    else:
        if dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root=dataset_dir['cifar10'], train=True, download=True, transform=CIFAR10_transform_train2)
            testset = torchvision.datasets.CIFAR10(root=dataset_dir['cifar10'], train=False, download=False, transform=CIFAR10_transform_test2)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                                    shuffle=False, num_workers=num_workers)
        if dataset == 'cifar100':
            trainset = torchvision.datasets.CIFAR10(root=dataset_dir['cifar100'], train=True, download=True, transform=CIFAR100_transform_train2)
            testset = torchvision.datasets.CIFAR10(root=dataset_dir['cifar100'], train=False, download=False, transform=CIFAR100_transform_test2)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                                    shuffle=False, num_workers=num_workers)
        if dataset == 'mnist': 
            trainset = torchvision.datasets.CIFAR10(root=dataset_dir['mnist'], train=True, download=True, transform=MNIST_transform_train2)
            testset = torchvision.datasets.CIFAR10(root=dataset_dir['mnist'], train=False, download=False, transform=MNIST_transform_test2)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                                    shuffle=False, num_workers=num_workers)
    num_class = num_classes[dataset]
    return trainloader, testloader, num_class, trainset, testset
        