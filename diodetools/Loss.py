import torch
from torchmetrics.functional import image_gradients, structural_similarity_index_measure
import torch.nn.functional as F 
import numpy as np 

def loss_l1(y_true, y_pred):
    calc = torch.mean(torch.abs(y_true - y_pred))
    # print(f"L1 : {calc.item()}")
    return calc

def loss_depthsmoothness(y_true, y_pred):
    dy_true, dx_true = image_gradients(y_true)
    dy_pred, dx_pred = image_gradients(y_pred)
    # dx_true, dy_true = image_gradients(y_true)
    # dx_pred, dy_pred = image_gradients(y_pred)
    
    weights_x = torch.exp(torch.mean(torch.abs(dx_true)))
    weights_y = torch.exp(torch.mean(torch.abs(dy_true)))

    # Depth smoothness
    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y
    
    depth_smoothness_loss = torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))
    # print(f"Depth smoothness : {depth_smoothness_loss.item()}")
    return depth_smoothness_loss

# def loss_ssim(y_true, y_pred): 
#     ssim = structural_similarity_index_measure(y_pred,  y_true, sigma=1.5, kernel_size=7, k1=0.01**2, k2=0.03**2)
#     return (1 - ssim)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([ np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def loss_ssim(img1, img2, max_val, kernel_size=11, size_average=True, full=False, k1=0.01, k2=0.03):
    padd = 0
    (batch, channel, height, width) = img1.size()
    
    real_size = min(kernel_size, height, width)
    window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # print(f"{mu1}\n{mu2}\n{mu1_mu2}")

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (k1 * max_val) ** 2
    C2 = (k2 * max_val) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    # print(f"L SSIM : {(1 - ret).item()}")
    # print()
    return torch.clamp((1 - ret), 0, 1)