import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from filters import low_pass, high_pass

@torch.no_grad()
def evaluate(model, loader, device, filter_fn=None, **filter_kwargs):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if filter_fn is not None:
            x = torch.stack([filter_fn(img, **filter_kwargs) for img in x])
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct/total

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10("data", train=False, download=True, transform=transform)
    loader = DataLoader(testset, batch_size=64, shuffle=False)
    model = models.resnet18(num_classes=10) #resnet18
    model.to(device)
    acc_clean = evaluate(model, loader, device)
    acc_low = evaluate(model, loader, device, filter_fn=low_pass, cutoff=15)
    acc_high = evaluate(model, loader, device, filter_fn=high_pass, cutoff=15)
    print(f"Clean acc: {acc_clean:.3f}")
    print(f"Low-pass acc: {acc_low:.3f}")
    print(f"High-pass acc: {acc_high:.3f}")

if __name__ == "__main__":
    main()
