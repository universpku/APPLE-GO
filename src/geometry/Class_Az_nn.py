import torch
import torch.nn as nn

class Az_nn_tensor(nn.Module):
    '''
    计算Gap_Az的神经网络
    '''
    def __init__(self,block_size_x,block_size_y,g=0.5):
        '''
        初始化函数，常规情况下block_size_x=block_size_y
        :param block_size_x: x方向的大像元包含的小像元数
        :param block_size_y: y方向的大像元包含的小像元数
        '''
        super(Az_nn_tensor, self).__init__()
        self.block_size_x=block_size_x
        self.block_size_y=block_size_y
        self.G = g


    def forward(self, input_tensor):
        '''
        计算Az的函数
        :param input_tensor: [3, H, W] 张量，通道顺序：CHM, PATH, FAVD
        :return: 聚合后的结果张量 [1, num_block_x, num_block_y]
        '''
        # 替换所有原地操作为安全操作
        input_tensor = torch.where(input_tensor < 0,
                                   torch.tensor(0.0, device=input_tensor.device),
                                   input_tensor)

        # 计算核心公式
        result_tensor = input_tensor[1] * input_tensor[2] * self.G * -1
        result_tensor = torch.exp(result_tensor)

        # 重构条件判断逻辑
        mask_chm = (input_tensor[0] > 0)
        result_tensor = torch.where(mask_chm,
                                    torch.tensor(0.0, device=input_tensor.device),
                                    result_tensor)

        mask_ge1 = (result_tensor >= 1)
        result_tensor = torch.where(mask_ge1,
                                    torch.tensor(0.0, device=input_tensor.device),
                                    result_tensor)

        # 计算校准张量（非原地版本）
        cal_tensor = torch.where(result_tensor > 0,
                                 torch.tensor(1.0, device=input_tensor.device),
                                 torch.tensor(0.0, device=input_tensor.device))

        # 分块聚合计算
        num_block_x = input_tensor.shape[1] // self.block_size_x
        num_block_y = input_tensor.shape[2] // self.block_size_y

        def block_aggregate(tensor):
            return tensor.view(
                1,
                num_block_x, self.block_size_x,
                num_block_y, self.block_size_y
            ).sum(dim=(2, 4))  # 同时聚合两个维度

        # 并行计算两个张量的分块聚合
        numerator = block_aggregate(result_tensor)
        denominator = block_aggregate(cal_tensor) + 1e-8

        # 最终结果计算
        result_tensor = numerator / denominator
        return result_tensor