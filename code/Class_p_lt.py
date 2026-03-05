import torch
import torch.nn as nn

class cal_P_lt(nn.Module):
    def __init__(self,size_x,g):
        super(cal_P_lt, self).__init__()
        self.size_x=size_x
        self.G=g


    def forward(self, input_tensor):
        '''
        :param input_tensor: [4, H, W] 张量，通道顺序：FAVD, sza, max_At, mean_At
        :return: 计算结果张量 [H, W]
        '''
        # 确保计算设备一致
        device = input_tensor.device

        # 生成高度序列（非原地版本）
        z_steps = torch.arange(
            1.0 / (self.size_x * 2),
            1,
            1.0 / self.size_x,
            device=device
        ).view(-1, 1, 1) * input_tensor[2]  # [N, H, W]

        # 预计算公共项
        cos_sza = torch.cos(input_tensor[1])  # [H, W]
        G_FAVD = self.G * input_tensor[0]  # [H, W]

        # 分步计算指数项（使用独立变量存储中间结果）
        part1_exp = torch.exp(z_steps * G_FAVD * (-1))  # exp(-G*FAVD*z)
        part2_exp = torch.exp(z_steps * G_FAVD * (-1) / cos_sza)  # exp(-G*FAVD*z/cosθ)
        part3_exp = torch.exp(-self.G * input_tensor[3] * input_tensor[0])  # [H, W]

        # 合并所有项（广播维度对齐）
        combined = part1_exp * part2_exp * part3_exp.unsqueeze(0)  # [N, H, W]

        # 聚合计算（保持维度）
        result = combined.sum(dim=0) * input_tensor[2] / combined.shape[0]

        return result
