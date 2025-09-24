import torch
import torch.nn as nn


class cal_P_lc(nn.Module):
    def __init__(self, size_x, g, bl):
        super(cal_P_lc, self).__init__()
        self.size_x = size_x
        self.G = g
        self.bL = bl

    def old_forward(self, input_tensor):
        '''

        :param input_tensor: 0:FAVD,1:sza，2：max_Ac
        :return:
        '''

        z_list = torch.arange(1.0 / self.size_x / 2, 1, 1.0 / self.size_x)
        z_list = z_list.reshape(-1, 1, 1)
        z_list = z_list * input_tensor[2, :, :]
        # print(z_list[:,0,0])
        cos_sza = torch.cos(input_tensor[1, :, :])
        tan_sza = torch.tan(input_tensor[1, :, :])
        part1 = z_list * self.G * input_tensor[0, :, :] * (-1)
        part2 = z_list * self.G * input_tensor[0, :, :] * (-1) / cos_sza
        part3 = z_list * tan_sza / self.bL * (-1)
        part4 = z_list * self.G * torch.sqrt(1 / cos_sza) * input_tensor[0, :, :]
        part1 = torch.exp(part1)
        part2 = torch.exp(part2)
        part3 = torch.exp(part3)
        part4 = torch.exp(part4 * part3)
        result_tensor = part1 * part2 * part4
        input_tensor[2, :, :] = input_tensor[2, :, :] + 1e-8
        result_tensor = result_tensor.sum(dim=(0)) / (result_tensor.shape[0] * input_tensor[2, :, :])
        input_tensor[2, :, :][input_tensor[2, :, :] > 1e-8] = 1
        input_tensor[2, :, :][input_tensor[2, :, :] <= 1e-8] = 0
        result_tensor = result_tensor * input_tensor[2, :, :]
        # print("input_tensor[2,1,2]",input_tensor[2,1,2])
        # print(1.0/1e-8)
        return result_tensor

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
        # part2 = torch.exp(z_grid * self.G * input_tensor[0] * (-1) / cos_sza)  # exp(-G*FAVD*z/cosθ)
        # part2 = torch.exp(z_grid * self.G * input_tensor[0] * (-1) )  # exp(-G*FAVD*z/cosθ)
        # part2 = torch.exp(z_grid * self.G * input_tensor[0] * (-1) / ((cos_sza+1)/2))  # exp(-G*FAVD*z/cosθ)
        # part2 = torch.exp(z_grid * self.G * input_tensor[0] * (-1)*1.1)  # exp(-G*FAVD*z/cosθ)
        # part2=1

        # test1
        # part2 = torch.exp(z_grid * self.G * input_tensor[0] * (-1))
        # test2
        # part2 = torch.exp(z_grid * self.G * input_tensor[0] * (-1)/ cos_sza*0.3)
        part2 = torch.exp(z_grid * self.G * input_tensor[0] * (-1) * cos_sza * 0.5)

        part3 = torch.exp(z_grid * tan_sza / self.bL * (-1))  # exp(-z*tanθ/bL)
        part4 = torch.exp(
            z_grid * self.G * torch.sqrt(1 / cos_sza) * input_tensor[0] * part3)  # exp(G*FAVD*z*sqrt(1/cosθ))
        # part4 = torch.exp(z_grid * self.G * torch.sqrt(1 / cos_sza)  * part3)  # exp(G*FAVD*z*sqrt(1/cosθ))
        # part4 = torch.exp(z_grid * self.G * torch.sqrt(1/ cos_sza / cos_sza) * input_tensor[0] * part3)  # exp(G*FAVD*z*sqrt(1/cosθ))

        # 尝试一下李的公式
        # part4 = torch.exp(self.G * torch.sqrt(1 / cos_sza)*input_tensor[0] * self.bL / (z_grid * tan_sza) * (1-part3))  # exp(G*FAVD*z*sqrt(1/cosθ))

        # 合并所有项
        combined = part1 * part2 * part4

        # 处理max_Ac（非原地版本）
        max_Ac_processed = input_tensor[2] + 1e-8  # 创建新张量代替原地操作

        # 计算归一化因子
        # normalization = combined.sum(dim=0) / (z_steps.size(0) * max_Ac_processed)
        normalization = combined.sum(dim=0) / (z_steps.size(0)) * input_tensor[2]

        # 生成掩码（非原地操作）
        mask = (input_tensor[2] > 1e-8).float()  # 直接生成0/1掩码

        # 最终结果
        result = normalization * mask
        return result

    def forward_test52(self, input_tensor, lrmean):
        '''
        :param input_tensor: [3, H, W] 张量，通道顺序：FAVD, sza, max_Ac
        :return: 计算结果张量 [H, W]
        '''
        # 生成高度序列（非原地操作）
        z_steps = torch.arange(1.0 / (self.size_x * 2), 1, 1.0 / self.size_x, device=input_tensor.device)
        z_grid = z_steps.view(-1, 1, 1) * input_tensor[2]  # [N, H, W]

        # 预计算三角函数
        cos_sza = torch.cos(input_tensor[1])  # [H, W]
        tan_sza = torch.tan(input_tensor[1])  # [H, W]

        # 分步计算指数项（保持中间变量）
        part1 = torch.exp(z_grid * self.G * input_tensor[0] * (-1))  # exp(-G*FAVD*z)
        # part2 = torch.exp(z_grid * self.G * input_tensor[0] * (-1) / cos_sza)  # exp(-G*FAVD*z/cosθ)
        # part2 = torch.exp(z_grid * self.G * input_tensor[0] * (-1) )  # exp(-G*FAVD*z/cosθ)
        # part2 = torch.exp(z_grid * self.G * input_tensor[0] * (-1) / ((cos_sza+1)/2))  # exp(-G*FAVD*z/cosθ)
        part2 = torch.exp(z_grid * self.G * input_tensor[0] * (-1) * lrmean)  # exp(-G*FAVD*z/cosθ)
        part3 = torch.exp(z_grid * tan_sza / self.bL * (-1))  # exp(-z*tanθ/bL)
        part4 = torch.exp(
            z_grid * self.G * torch.sqrt(1 / cos_sza) * input_tensor[0] * part3)  # exp(G*FAVD*z*sqrt(1/cosθ))
        # part4 = torch.exp(z_grid * self.G * torch.sqrt(1 / cos_sza)  * part3)  # exp(G*FAVD*z*sqrt(1/cosθ))

        # 尝试一下李的公式
        # part4 = torch.exp(self.G * torch.sqrt(1 / cos_sza)*input_tensor[0] * self.bL / (z_grid * tan_sza) * (1-part3))  # exp(G*FAVD*z*sqrt(1/cosθ))

        # 合并所有项
        combined = part1 * part2 * part4

        # 处理max_Ac（非原地版本）
        max_Ac_processed = input_tensor[2] + 1e-8  # 创建新张量代替原地操作

        # 计算归一化因子
        # normalization = combined.sum(dim=0) / (z_steps.size(0) * max_Ac_processed)
        normalization = combined.sum(dim=0) / (z_steps.size(0)) * input_tensor[2]

        # 生成掩码（非原地操作）
        mask = (input_tensor[2] > 1e-8).float()  # 直接生成0/1掩码

        # 最终结果
        result = normalization * mask
        return result
