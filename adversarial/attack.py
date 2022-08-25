import torchattacks
from torchvision.models import resnet18
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from visdom import Visdom

def PGD(images, labels, model, eps=0.3, alpha=2/255, steps=40):
    atk = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    return atk(images, labels)


# atk.set_mode_targeted_least_likely(kth_min)  # Targeted attack
# atk.set_return_type(type='int')  # Return values [0, 255]
# atk = torchattacks.MultiAttack([atk1, ..., atk99])  # Combine attacks
# atk.save(data_loader, save_path=None, verbose=True, return_verbose=False)  # Save adversarial images

if __name__ == "__main__":
    print("running...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])
    trainset = datasets.MNIST(root='../Dataset/',download=True ,train=True, transform=transform)
    
    
