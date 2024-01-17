import torch
from skimage.metrics import structural_similarity
import cv2
import numpy as np
def PSNR(original, compressed): 
    mse = torch.mean((original - compressed) ** 2) 
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr 

def mae(original, compressed):
    return torch.mean(torch.abs(original - compressed))

def SSIM(original, reconstructed):
    data_range = 1.0
    ssim_index, _ = structural_similarity(original, reconstructed, full=True, data_range=data_range, win_size=5)
    return ssim_index