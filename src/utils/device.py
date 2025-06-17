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
