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
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CIFAR():
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean[dataset], std[dataset]),
        ]) # meanstd transformation

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean[dataset], std[dataset]),
        ])

        self.trainlength = 0
        self.testlength = 0

    def length(self):
        return len(self.trainset), len(self.testset)

    def dataloader(self, batch_size):
        if(self.dataset == 'cifar10'):
            trainset = torchvision.datasets.CIFAR10(root=dataset_dir[self.dataset], train=True, download=True, transform=self.transform_train)
            testset = torchvision.datasets.CIFAR10(root=dataset_dir[self.dataset], train=False, download=False, transform=self.transform_test)
            num_classes = 10
        elif(self.dataset == 'cifar100'):
            trainset = torchvision.datasets.CIFAR100(root=dataset_dir[self.dataset], train=True, download=True, transform=self.transform_train)
            testset = torchvision.datasets.CIFAR100(root=dataset_dir[self.dataset], train=False, download=False, transform=self.transform_test)
            num_classes = 100

        self.trainlength = len(trainset)
        self.testlength = len(testset)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
        return trainloader, testloader, num_classes

class Mnist(object):
    def __init__(self) -> None:
        self.transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
        self.trainlength = 0
        self.testlength = 0 

    def dataset(self):
        trainset = torchvision.datasets.MNIST(root=dataset_dir['mnist'], train=True, download=True, transform=self.transform)
        testset = torchvision.datasets.MNIST(root=dataset_dir['mnist'], train=False, download=True, transform=self.transform)
        self.trainlength = len(trainset)
        self.testlength = len(testset)
        return trainset, testset
    
    def dataloader(self, batch_size):
        trainset,testset = self.dataset()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_worker=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_worker=4)
        return trainloader, testloader


if __name__ == "__main__":
    pass