import torch.nn as nn
import torch.nn.functional as F

class DynamicBlockUpsample2D(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        # 输入 x 形状: [batch, channels, h, w]
        h, w = x.shape[2], x.shape[3]
        x_upsampled = F.interpolate(
            x,
            scale_factor=self.scale_factor,  # 动态放大倍数
            mode='nearest'
        )
        return x_upsampled