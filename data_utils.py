from torchvision import datasets, transforms

def get_cifar10(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.CIFAR10("data", train=True, download=True, transform=transform) # CIFAR-10 dataset:32x32 
    test = datasets.CIFAR10("data", train=False, download=True, transform=transform)
    return train, test 
