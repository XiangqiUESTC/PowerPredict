from core.base_processor import BaseProcessor
import random
import torch
from torchvision.ops import roi_align


class RoIAlign(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.feature_map = None
        self.rois = None
        self.aligned = None

        # 递增模式初始化
        if self.mode == "increase":
            self.times = 0  # 运行次数计数器

    def generate_config(self):
        if self.mode == "random":
            # 使用yaml配置参数
            B = random.randint(self.min_B, self.max_B)
            C = random.randint(self.min_C, self.max_C)
            H = random.randint(self.min_H, self.max_H)
            W = random.randint(self.min_W, self.max_W)
            num_rois = random.randint(self.min_num_rois, self.max_num_rois)
            pooled_h = random.randint(self.min_pooled_size, self.max_pooled_size)
            pooled_w = random.randint(self.min_pooled_size, self.max_pooled_size)
            spatial_scale = round(random.uniform(self.spatial_scale_min, self.spatial_scale_max), 2)
            sampling_ratio = random.choice(self.sampling_ratio_options)
            aligned = random.choice(self.aligned_options)

        elif self.mode == "increase":
            # 线性递增参数
            B = min(self.base_B + self.times, self.max_B)
            C = min(self.base_C + self.times * 2, self.max_C)  # 通道数增长更快
            H = min(self.base_H + self.times * 8, self.max_H)  # 高度增长更快
            W = min(self.base_W + self.times * 8, self.max_W)  # 宽度增长更快
            num_rois = min(self.base_num_rois + self.times * 4, self.max_num_rois)

            # 固定其他参数
            pooled_h, pooled_w = self.output_size
            spatial_scale = self.spatial_scale
            sampling_ratio = self.sampling_ratio
            aligned = self.aligned

            self.times += 1  # 更新计数器

        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        self.config = {
            "feature_map_shape": [B, C, H, W],
            "num_rois": num_rois,
            "output_size": (pooled_h, pooled_w),
            "spatial_scale": spatial_scale,
            "sampling_ratio": sampling_ratio,
            "aligned": aligned,
            "device": self.device,
        }
        return self.config

    def setup(self):
        B, C, H, W = self.config["feature_map_shape"]
        num_rois = self.config["num_rois"]
        self.output_size = self.config["output_size"]
        self.spatial_scale = self.config["spatial_scale"]
        self.sampling_ratio = self.config["sampling_ratio"]
        self.aligned = self.config["aligned"]

        # 特征图 (直接创建在目标设备上)
        self.feature_map = torch.randn(B, C, H, W, device=self.device)

        # 随机生成 ROI：[batch_idx, x1, y1, x2, y2]
        rois = []
        for _ in range(num_rois):
            batch_idx = random.randint(0, B - 1)
            x1 = random.uniform(0, W * 0.8)
            y1 = random.uniform(0, H * 0.8)
            x2 = random.uniform(x1 + 1, W)
            y2 = random.uniform(y1 + 1, H)
            rois.append([batch_idx, x1, y1, x2, y2])

        # 创建ROI张量并移动到目标设备
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