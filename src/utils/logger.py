import logging
import os.path
from datetime import datetime
from os.path import join, dirname


class Logger:
    def __init__(self, log_dir, test_name="Test"):
        # 初始化日志实例对象
        logger = logging.getLogger(f"{test_name} Log")
        logger.setLevel(logging.DEBUG)

        # 设置控制台输出handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # 日志文件绝对路径
        file_path = join(log_dir, f'{test_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')

        # 设置文件输出handler
        os.makedirs(dirname(file_path), exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # 格式化输出
        formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s： %(message)s', '%H:%M:%S')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # 添加handler
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        self.logger = logger

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)