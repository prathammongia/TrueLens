import torch

def fft_transform(img_tensor):
    img_gray = img_tensor.mean(dim=0)
    fft = torch.fft.fft2(img_gray)
    fft = torch.abs(fft)
    fft = torch.log(fft + 1e-8)
    return fft.unsqueeze(0)
