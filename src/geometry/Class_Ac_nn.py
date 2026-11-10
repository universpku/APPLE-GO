import torch
import torch.nn as nn

class Ac_nn_tensor(nn.Module):
    '''
    计算Ac的神经网络
    '''
    def __init__(self,block_size_x,block_size_y,g=0.5):
        '''
        初始化函数，常规情况下block_size_x=block_size_y
        :param block_size_x: x方向的大像元包含的小像元数
        :param block_size_y: y方向的大像元包含的小像元数
        '''
        super(Ac_nn_tensor, self).__init__()
        self.block_size_x=block_size_x
        self.block_size_y=block_size_y
        self.G = g


    def forward(self, input_tensor):
        '''
        计算Ac的函数
        :param input_tensor: [6, H, W] 张量，通道顺序：CHM, PATH, FAVD, Edge, TH, Max_H
        :return: [3, num_block_x, num_block_y] 矩阵，通道：count_Ac, GAP_Vertical_Ac, Max_Ac
        '''
        # 消除所有原地操作，保持计算图完整
        num_block_x = input_tensor.shape[1] // self.block_size_x
        num_block_y = input_tensor.shape[2] // self.block_size_y

        # 处理输入张量（非原地版本）
        input_clean = torch.where(input_tensor < 0,
                                  torch.tensor(0.0, device=input_tensor.device),
                                  input_tensor)

        # 构建result_tensor的三个通道
        # 通道1计算
        ch1_base_0 = input_clean[0] - input_clean[4]
        ch1_base = torch.where(ch1_base_0<0,
                               torch.tensor(0.0, device=input_tensor.device),
                                 ch1_base_0)  # 确保ch1_base非负

        deleteNumber=ch1_base[ch1_base > 0]
        if deleteNumber.numel() > 0:
            min_positive = torch.min(deleteNumber)
        else:
            min_positive = 0    

        ch1 = (ch1_base-min_positive) * input_clean[2] * self.G * (-1)
        ch1 = torch.exp(ch1)

        # 通道2计算
        ch2_base = input_clean[0] - input_clean[4]

        # 通道0计算逻辑重构
        mask_1to4 = (input_clean[1:4] > 0)
        modified_1to4 = torch.where(mask_1to4,
                                    torch.tensor(-1.0, device=input_tensor.device),
                                    input_clean[1:4])
        modified_1to4 += 1  # 此时操作不会影响原始输入

        ch0_mask = input_clean[0]  * modified_1to4[2]
        ch0_mask = torch.where(ch0_mask > 0,
                          torch.tensor(1.0, device=input_tensor.device),
                          torch.tensor(0.0, device=input_tensor.device))
        path_mask=torch.exp(input_clean[1] * input_clean[2] * self.G * (-1))
        ch0= ch0_mask * path_mask

        # 构建完整result_tensor
        result_tensor = torch.stack([
            ch0,  # 通道0
            ch1 * ch0,  # 通道1（应用掩码）
            ch2_base * ch0  # 通道2（应用掩码）
        ], dim=0)

        # 分块聚合计算
        def block_aggregate(tensor, mode='sum'):
            view_shape = (1, num_block_x, self.block_size_x, num_block_y, self.block_size_y)
            if mode == 'sum':
                return tensor.view(view_shape).sum(dim=(2, 4))
            elif mode == 'max':
                return tensor.view(view_shape).amax(dim=(2, 4))

        # 并行计算各通道聚合
        ch0_agg = block_aggregate(result_tensor[0], 'sum')  # count_Ac
        ch1_agg = block_aggregate(result_tensor[1], 'sum')  # 分子
        ch2_agg = block_aggregate(result_tensor[2], 'max')  # Max_Ac
        ch3_agg = block_aggregate(result_tensor[2], 'sum')  # Max_Ac

        # 构建最终输出
        result_output = torch.stack([
            ch0_agg,  # 通道0
            ch1_agg / (ch0_agg + 1e-8),  # 通道1（GAP_Vertical_Ac）
            ch2_agg  # 通道2
            # ch3_agg/ (ch0_agg + 1e-8)
        ], dim=0)

        return result_output
