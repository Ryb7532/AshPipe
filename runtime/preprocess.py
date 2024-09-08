import torch


def cutout(x, y, pad_size, prob=1.0):
    B, C, H, W = x.shape
    center_h, center_w = torch.randint(high=H, size=(1,)), torch.randint(high=W, size=(1,))
    low_h, high_h = torch.clamp(center_h-pad_size, 0, H).item(), torch.clamp(center_h+pad_size, 0, H).item()
    low_w, high_w = torch.clamp(center_w-pad_size, 0, W).item(), torch.clamp(center_w+pad_size, 0, W).item()
    mask = torch.ones((H, W)).cuda()
    mask[low_h:high_h, low_w:high_w] = 0
    return mask * x if torch.rand(1)<=prob else x, y
