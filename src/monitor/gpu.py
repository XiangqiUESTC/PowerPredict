import torch

from utils.device import is_device_avail_on_torch

def get_gpu_info(device):
    pass

def get_gpu_model(device):
    """

        argument:
            device: 字符串
        return:
            gpu_model: gpu型号名，如果device不合法则是None
    """
    # device应该是字符串
    if not isinstance(device, str):
        raise TypeError("function get_gpu_model expect to receive param device of type:str")

    if device.startswith("cuda"):
        return torch.cuda.get_device_name(device)
    else:
        return None
