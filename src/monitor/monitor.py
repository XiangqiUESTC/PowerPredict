"""
    包括主要的监测逻辑函数和监控线程
"""
from operators import *
from models import *

# 同时注册算子和模型
REGISTRY = {
    **OPERATOR_REGISTRY,
    **MODEL_REGISTRY,
}

def run_and_monitor(args, logger, op_names, num_samples):
    # 记录开始时间

    # 开始主循环
    for op_name in op_names:
        op = REGISTRY[op_name](args, logger)
        logger.info(f"对{op_name}的实验开始!测试{num_samples}次!")
        logger.info(f"默认设备是{op.device}")
        operation_monitor(
            op,
            op_name,
            logger,
            num_samples,
        )
        logger.info(f"对算子{op_name}的{num_samples}次测试结束!")
        logger.info(
            "--------------------------------------------------------------------------------------------------------------------------------------")  # 分割

    logger.info("实验结束！")

def operation_monitor(a, b, c, d):
    pass

def cpu_monitor_thread():
    pass

def gpu_monitor_thread():
    pass

def disk_monitor_thread():
    pass

def memory_monitor_thread():
    pass