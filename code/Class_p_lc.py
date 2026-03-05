import torch
import torch.nn as nn


class cal_P_lc(nn.Module):
    def __init__(self, size_x, g, bl):
        super(cal_P_lc, self).__init__()
        self.size_x = size_x
        self.G = g
        self.bL = bl


    def forward(self, input_tensor):
        '''
        :param input_tensor: [3, H, W] 张量，通道顺序：FAVD, sza, max_Ac
        :return: 计算结果张量 [H, W]
        '''
        # input_tensor[2]=input_tensor[2]
        # 生成高度序列（非原地操作）
        z_steps = torch.arange(1.0 / (self.size_x * 2), 1, 1.0 / self.size_x, device=input_tensor.device)
        z_grid = z_steps.view(-1, 1, 1) * input_tensor[2]  # [N, H, W]

        # 预计算三角函数
        cos_sza = torch.cos(input_tensor[1])  # [H, W]
        tan_sza = torch.tan(input_tensor[1])  # [H, W]

        # 分步计算指数项（保持中间变量）
        part1 = torch.exp(z_grid * self.G * input_tensor[0] * (-1))  # exp(-G*FAVD*z)
        part2 = torch.exp(z_grid * self.G * input_tensor[0] * (-1) * cos_sza * 0.5)
        part3 = torch.exp(z_grid * tan_sza / self.bL * (-1))  # exp(-z*tanθ/bL)
        part4 = torch.exp(
            z_grid * self.G * torch.sqrt(1 / cos_sza) * input_tensor[0] * part3)  # exp(G*FAVD*z*sqrt(1/cosθ))
        
        # 合并所有项
        combined = part1 * part2 * part4

        # 处理max_Ac（非原地版本）
        nan_mask=torch.isnan(combined)
        combined = combined.masked_fill(nan_mask, 0.0)

        # 计算归一化因子
        normalization = combined.sum(dim=0) / (z_steps.size(0)) * input_tensor[2]

        # 生成掩码（非原地操作）
        mask = (input_tensor[2] > 1e-8).float()  # 直接生成0/1掩码

        # 最终结果
        result = normalization * mask

        return result

