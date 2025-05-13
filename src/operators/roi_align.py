from core.base_processor import BaseProcessor
import random
import torch

from torchvision.ops import roi_align


class RoIAlign(BaseProcessor):
    def __init__(self, logger):
        super().__init__(logger)
        self.feature_map = None
        self.rois = None
        self.output_size = None
        self.spatial_scale = None
        self.sampling_ratio = None
        self.aligned = None

    def generate_config(self):
        # 图像特征 map 的参数
        B = random.randint(1, 8)  # batch size
        C = random.randint(1, 256)  # channels
        H = random.randint(32, 128)  # height
        W = random.randint(32, 128)  # width
        num_rois = random.randint(1, 64)  # ROI 数量
        pooled_h = random.randint(2, 14)
        pooled_w = random.randint(2, 14)
        self.config = {
            "feature_map_shape": [B, C, H, W],
            "num_rois": num_rois,
            "output_size": (pooled_h, pooled_w),
            "spatial_scale": round(random.uniform(0.1, 1.0), 2),
            "sampling_ratio": random.choice([-1, 0, 1, 2]),
            "aligned": random.choice([True, False])
        }
        return self.config

    def setup(self):
        B, C, H, W = self.config["feature_map_shape"]
        num_rois = self.config["num_rois"]
        self.output_size = self.config["output_size"]
        self.spatial_scale = self.config["spatial_scale"]
        self.sampling_ratio = self.config["sampling_ratio"]
        self.aligned = self.config["aligned"]

        # 特征图
        self.feature_map = torch.randn(B, C, H, W, device=self.device)

        # 随机生成 roi：[batch_idx, x1, y1, x2, y2]
        rois = []
        for _ in range(num_rois):
            batch_idx = random.randint(0, B - 1)
            x1 = random.uniform(0, W * 0.8)
            y1 = random.uniform(0, H * 0.8)
            x2 = random.uniform(x1 + 1, W)
            y2 = random.uniform(y1 + 1, H)
            rois.append([batch_idx, x1, y1, x2, y2])
        self.rois = torch.tensor(rois, dtype=torch.float32, device=self.device)

    def execute(self):
        return roi_align(
            input=self.feature_map,
            boxes=self.rois,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned
        )
