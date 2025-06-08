from core.base_processor import BaseProcessor
import random
import torch

from torchvision.ops import nms


class NMS(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.boxes = None
        self.scores = None
        self.iou_threshold = None

        if self.mode == "increase":
            self.times = 0
            self.num_boxes = self.start_num_boxes
            self.max_boxes = self.max_boxes

    def generate_config(self):
        if self.mode == "random":
            num_boxes = random.randint(self.min_boxes, self.max_boxes)
            iou_threshold = round(random.uniform(self.min_iou_threshold, self.max_iou_threshold), 2)

        elif self.mode == "increase":
            num_boxes = min(self.num_boxes + self.times * self.step_increment, self.max_boxes)
            iou_threshold = self.base_iou_threshold + self.times * self.iou_increment
            iou_threshold = min(round(iou_threshold, 2), 1.0)
            self.times += 1

        else:
            raise NotImplementedError

        self.config = {
            "num_boxes": num_boxes,
            "iou_threshold": iou_threshold,
            "device": self.device,
        }
        return self.config

    def setup(self):
        num_boxes = self.config["num_boxes"]
        self.iou_threshold = self.config["iou_threshold"]

        # 随机生成合法的 boxes: [x1, y1, x2, y2] 且 x2 > x1, y2 > y1
        x1 = torch.rand(num_boxes) * 512
        y1 = torch.rand(num_boxes) * 512
        x2 = x1 + torch.rand(num_boxes) * 50 + 1
        y2 = y1 + torch.rand(num_boxes) * 50 + 1
        self.boxes = torch.stack([x1, y1, x2, y2], dim=1).to(self.device)

        self.scores = torch.rand(num_boxes).to(self.device)

    def execute(self):
        return nms(self.boxes, self.scores, self.iou_threshold)
