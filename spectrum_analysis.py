import torch
import torch.fft as fft
import matplotlib.pyplot as plt

def radial_spectrum(img):
    C, H, W = img.shape
    f = fft.fft2(img)
    fshift = fft.fftshift(f)
    power = fshift.abs() ** 2
    power = power.mean(0)
    cx, cy = H // 2, W // 2
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    dist = torch.sqrt((X - cy) ** 2 + (Y - cx) ** 2).long()
    max_r = dist.max().item()
    radial = torch.zeros(max_r + 1)
    counts = torch.zeros(max_r + 1)
    for r in range(max_r + 1):
        mask = (dist == r)
        radial[r] = power[mask].sum()
        counts[r] = mask.sum()
    return (radial / counts.clamp(min=1)).cpu()

def plot_spectrum(img):
    radial = radial_spectrum(img)
    plt.semilogy(radial.numpy())
    plt.xlabel("Frequency radius")
    plt.ylabel("Power (log scale)")
    plt.title("Radially averaged power spectrum")
    plt.show()
