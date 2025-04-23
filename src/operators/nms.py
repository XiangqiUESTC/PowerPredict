from core.base_processor import BaseProcessor
import random
import torch

from torchvision.ops import nms


class NMS(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.boxes = None
        self.scores = None
        self.iou_threshold = None

    def generate_config(self):
        MAX_BOXES = 1000
        MIN_BOXES = 10

        num_boxes = random.randint(MIN_BOXES, MAX_BOXES)
        iou_threshold = round(random.uniform(0.3, 0.7), 2)  # 比较常见的阈值范围

        self.config = {
            "num_boxes": num_boxes,
            "iou_threshold": iou_threshold
        }
        return self.config

    def setup(self):
        num_boxes = self.config["num_boxes"]
        self.iou_threshold = self.config["iou_threshold"]

        # 随机生成合法的 boxes: [x1, y1, x2, y2] 且 x2 > x1, y2 > y1
        x1 = torch.rand(num_boxes) * 512
        y1 = torch.rand(num_boxes) * 512
        x2 = x1 + torch.rand(num_boxes) * 50 + 1  # 保证 x2 > x1
        y2 = y1 + torch.rand(num_boxes) * 50 + 1  # 保证 y2 > y1
        self.boxes = torch.stack([x1, y1, x2, y2], dim=1).to("cuda")

        # 随机置信度分数
        self.scores = torch.rand(num_boxes).to("cuda")

    def execute(self):
        # 返回的是保留框的索引
        keep_indices = nms(self.boxes, self.scores, self.iou_threshold)
        return keep_indices
