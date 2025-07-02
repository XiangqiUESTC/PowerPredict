import torch

def is_device_avail_on_torch(device):
    if not isinstance(device, str):
        raise TypeError("设备应该是一个字符串！")
    if device == "cpu":
        return True
    elif device.startswith("cuda"):
        if not torch.cuda.is_available():
            return False
        else:
            if device == "cuda":
                return True
            elif ":" in device:
                try:
                    # 提取冒号后的数字部分
                    device_num = int(device.split(':')[1])
                    if device_num >= torch.cuda.device_count():
                        return False
                    else:
                        return True
                except (ValueError, IndexError):
                    return False
            else:
                return False
    elif device == "xpu":
        return hasattr(torch, 'xpu') and torch.xpu.is_available()
    elif device == "npu":
        return hasattr(torch, 'npu') and torch.npu.is_available()
    else:
        return False


def is_device_gpu(device):
    """
    Description
        判定一个字符串是不是gpu设备
    Arguments
        device: 字符串，类似'cpu', 'cuda:x', 'xpu', 'npu'等
    Return
        True or False
    """
    if not isinstance(device, str):
        raise TypeError("function is_device_gpu expect str type")

    device_lower = device.lower()

    # 检查是否以'cuda'开头（如'cuda:0'或'cuda'），或是'xpu'/'npu'
    return device_lower.startswith('cuda') or device_lower in ['xpu', 'npu']