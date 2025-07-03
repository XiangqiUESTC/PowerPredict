import torch
import subprocess

from utils.device import is_device_avail_on_torch
from utils.device import is_device_gpu

def get_gpu_info(device):
    """
        Arguments
            device: 不同的设备类型
        Return
            Dict={

            }
    """
    if not isinstance(device, str):
        raise TypeError("设备应该是一个字符串！")

    # 如果不是gpu设备返回None
    if not is_device_gpu(device):
        return None

    if device.startswith("cuda"):
        # 默认使用第一个设备
        device_num = 0
        # 从device字符串中提取设备编号
        if ':' in device:
            try:
                # 提取冒号后的数字部分
                device_num = int(device.split(':')[1])
            except (ValueError, IndexError):
                # 处理无效的设备编号格式
                print(f"警告: 无效的设备格式 '{device}'，默认使用设备0")
                device_num = 0
        try:
            output = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu',
                '--format=csv,noheader,nounits'
            ])
            infos = output.decode('utf-8').strip().splitlines()
            info = infos[device_num].split(',')
            return {
                "gpu_power": float(info[0]),
                "gpu_utilization": float(info[1]),
                "gpu_memory_used": float(info[2]),
                "gpu_temperature": float(info[3]),
            }
        except subprocess.CalledProcessError as e:
            return None

    elif device == "npu":
        try:
            # 执行 npu-smi info 命令获取原始输出
            output = subprocess.check_output("npu-smi info", shell=True, text=True)

            # 解析输出，提取所有NPU的功耗值
            power_values = []
            for line in output.split('\n'):
                # print("start get_gpu_power")
                # print("-" * 50)
                # print("-" * 50)
                # print(line)
                # print("-" * 50)
                # print("-" * 50)
                if '| 0     ' in line:  # 匹配NPU信息行
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    # print("parts :", parts)
                    # if len(parts) >= 4:
                    if parts[1] == 'OK':
                        power = parts[2].split()[0]  # 提取功耗值(如"93.9")
                        power_values.append(float(power))
                        # print("gpu功耗: ", power)

            # 计算平均功耗(如果有多个NPU)
            if power_values:
                return sum(power_values) / len(power_values)
            return None
        except Exception as e:
            print(f"获取NPU功耗出错: {e}")
            return None
    elif device == "xpu":
        return None
    else:
        raise Exception("Unknown device")

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
