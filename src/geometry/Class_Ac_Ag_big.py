import torch
import torch.nn as nn

class Ac_Ag_big_nn_tensor(nn.Module):
    '''
    计算Ac_big的神经网络
    '''
    def __init__(self,block_size_x,block_size_y,g=0.5):
        '''
        初始化函数，常规情况下block_size_x=block_size_y
        :param block_size_x: x方向的大像元包含的小像元数
        :param block_size_y: y方向的大像元包含的小像元数
        '''
        super(Ac_Ag_big_nn_tensor, self).__init__()
        self.block_size_x=block_size_x
        self.block_size_y=block_size_y
        self.G = g


    def forward(self, input_tensor):
        '''
        计算Ac_big的函数
        :param input_tensor: [4, H, W] 张量，通道顺序：CHM, PATH, FAVD, PATH2
        :return: [2, num_block_x, num_block_y] 张量，通道：Ac, Ag
        '''
        # 非原地处理输入张量
        # 步骤1：处理负值和PATH通道
        input_clean = torch.where(input_tensor < 0,
                                  torch.tensor(0.0, device=input_tensor.device),
                                  input_tensor)

        # 步骤2：处理PATH通道的条件逻辑
        path_modified = torch.where(input_clean[1] > 0,
                                    torch.tensor(-1.0, device=input_tensor.device),
                                    input_clean[1])
        path_modified += 1  # 非原地加法
        # path_modified 为path为0的部分位置

        # 步骤3：计算result_tensor的第一个通道
        ch0 = input_clean[0] * path_modified

        # 步骤4：处理CHM通道的条件逻辑
        chm_modified = torch.where(input_clean[0] > 0,
                                   torch.tensor(-1.0, device=input_tensor.device),
                                   input_clean[0])
        chm_modified += 1  # 非原地加法

        # 步骤5：计算result_tensor的第二个通道
        ch1 = chm_modified * path_modified

        # 步骤6：判断逻辑
        tensor = torch.stack([ch0, ch1], dim=0)  # [2,H,W]
        mid_tensor = torch.where(tensor > 0, 1.0, 0.0)  # 独立处理每个通道

        # 步骤7：核心公式计算
        result_tensor = mid_tensor * input_clean[2] * self.G * (-1) * input_clean[3]
        result_tensor = torch.exp(result_tensor) * mid_tensor

        # 分块聚合计算
        num_block_x = input_clean.shape[1] // self.block_size_x
        num_block_y = input_clean.shape[2] // self.block_size_y

        def block_aggregate(tensor):
            return tensor.view(
                2,  # 保持双通道结构
                num_block_x, self.block_size_x,
                num_block_y, self.block_size_y
            ).sum(dim=(2, 4))  # 聚合空间维度



        # 分子分母计算
        numerator = block_aggregate(result_tensor)
        denominator = block_aggregate(mid_tensor) + 1e-8


        mid_tensor2 = block_aggregate(mid_tensor)
        mid_tensor3= torch.where(mid_tensor2 > 0, torch.tensor(0.0, device=input_tensor.device), torch.tensor(1.0, device=input_tensor.device))
        numerator2= numerator + mid_tensor3
        denominator2= denominator + mid_tensor3

        # 最终结果
        # return numerator / denominator
        return numerator2 / denominator2