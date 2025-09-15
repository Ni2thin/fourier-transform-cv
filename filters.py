import torch
import torch.fft as fft

def _make_mask(shape, cutoff_low=0, cutoff_high=None, device="cpu"):
    H, W = shape
    cx, cy = H // 2, W // 2
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    dist = torch.sqrt((X - cy) ** 2 + (Y - cx) ** 2).to(device)
    if cutoff_high is None:
        cutoff_high = max(H, W)
    mask = (dist >= cutoff_low) & (dist <= cutoff_high)
    return mask.float()

def apply_filter(img, cutoff_low=0, cutoff_high=None):
    C, H, W = img.shape
    f = fft.fft2(img)
    f_shift = fft.fftshift(f)
    mask = _make_mask((H, W), cutoff_low, cutoff_high, img.device)
    mask = mask.unsqueeze(0).repeat(C, 1, 1)
    f_filtered = f_shift * mask
    f_ishift = fft.ifftshift(f_filtered)
    img_filtered = fft.ifft2(f_ishift).real
    return img_filtered
  
def low_pass(img, cutoff):
    return apply_filter(img, 0, cutoff)

def high_pass(img, cutoff):
    return apply_filter(img, cutoff, None)
