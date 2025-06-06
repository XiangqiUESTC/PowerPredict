import torch


def is_device_avail_on_torch(device):
    if device == "cpu":
        return True
    elif device == "cuda":
        return torch.cuda.is_available()
    elif device == "xpu":
        return hasattr(torch, 'xpu') and torch.xpu.is_available()
    elif device == "npu":
        return hasattr(torch, 'npu') and torch.npu.is_available()
    else:
        return False
