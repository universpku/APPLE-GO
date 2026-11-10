import torch
import torch.nn as nn

class cal_LAI(nn.Module):
    '''
    计算LAI的神经网络
    '''
    def __init__(self,block_size_x,block_size_y,g=0.5):
        super(cal_LAI, self).__init__()
        self.block_size_x=block_size_x
        self.block_size_y=block_size_y
        self.G = g


    def forward(self, input_tensor):
        '''
        计算LAI的函数
        :param input_tensor: [6, H, W] 张量，通道顺序：CHM, PATH, FAVD, Edge, TH, Max_H
        :return: [1, num_block_x, num_block_y] 张量，LAI
        '''
        # 非原地处理输入
        input_clean = torch.where(input_tensor < 0, 0.0, input_tensor)

        # 计算三个核心通道
        # 通道0: PATH_Vertical_Array_Ac
        ch0 = input_clean[0] - input_clean[4]

        # 通道1: PATH_Vertical_Array_At
        ch1 = input_clean[0] - input_clean[4]

        # 通道2: 复合逻辑计算
        # 步骤1: 初始化mask
        mask_path = (input_clean[1] != 0)
        mask_edge = (input_clean[3] == 1)
        ch2_mask = torch.where(mask_path | mask_edge, 0.0, 1.0)
        # return ch2_mask

        # 步骤2: CHM处理
        modified_chm = torch.where(input_clean[0] > 0, 1.0, input_clean[0])

        # 步骤3: 最终通道计算
        ch2 = (ch0 * ch2_mask) + (ch1 * (1 - ch2_mask))
        ch2 = ch2 * modified_chm * input_clean[2]

        # 分块聚合计算
        num_block_x = input_clean.shape[1] // self.block_size_x
        num_block_y = input_clean.shape[2] // self.block_size_y

        def block_aggregate(tensor, mode='sum'):
            view_shape = (1, num_block_x, self.block_size_x, num_block_y, self.block_size_y)
            reshaped = tensor.view(view_shape)
            if mode == 'sum':
                return reshaped.sum(dim=(2, 4))
            elif mode == 'mean':
                return reshaped.mean(dim=(2, 4))
            return reshaped

        # 分子部分计算
        numerator = block_aggregate(ch2, 'sum')

        # 分母部分计算
        edge_agg = block_aggregate(input_clean[3], 'sum')
        chm_agg = block_aggregate(modified_chm, 'sum')
        denominator = chm_agg + 1e-8

        # 最终结果计算
        result = (numerator / self.block_size_x / self.block_size_y) * \
                 ((-edge_agg / 2) + chm_agg) / denominator


        return result  # 保持输出形状[1, N, M]
