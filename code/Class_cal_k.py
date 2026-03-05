import torch
import torch.nn as nn

class cal_k(nn.Module):
    '''
    从Ac、Ag、At、Ac_Ag中计算k的神经网络
    '''
    def __init__(self,block_size_x,block_size_y):
        super(cal_k, self).__init__()
        self.block_size_x=block_size_x
        self.block_size_y=block_size_y


    def forward(self, input_tensor):
        '''
        计算k的函数
        :param input_tensor: [9, H, W] 张量，各通道含义见注释
        input_tensor: 是一个[9,n,n]的张量
                                0:Count_Ag
                                1:Gap_Az
                                2:Count_Ac
                                3:GAP_Vertical_Ac
                                4:GAP_Vertical_At
                                5:Gap_big_Ac
                                6:Gap_big_Ag
                                7:Count_edge
                                8:Gap_count
                                9:sza
        :return: [8, H, W] 张量，包含8种k值计算结果
        '''
        # 预计算公共项
        cos_sza= torch.cos(input_tensor[9] * torch.pi / 180)
        edge_term = input_tensor[7] * torch.exp(-cos_sza) / 2


        grid_size = self.block_size_x * self.block_size_y
        # 独立计算每个通道
        # 通道0-3：基础k值
        kc = (input_tensor[2] + edge_term) * (1 - input_tensor[3]) / grid_size
        kt = (grid_size - input_tensor[8] - input_tensor[2] - edge_term) * (1 - input_tensor[4]) / grid_size
        kz = (input_tensor[8] - input_tensor[0] - edge_term) * (1 - input_tensor[1]) / grid_size
        kg = 1 - kc - kt - kz


        # 通道4-7：big k值
        kc_big = kc * input_tensor[5]
        kt_big = kt + kc * (1 - input_tensor[5])
        kz_big = kz + kg * (1 - input_tensor[6])
        kg_big = 1 - kc_big - kt_big - kz_big

        # 合并所有结果
        result_tensor = torch.stack([
            kc, kt, kz, kg,
            kc_big, kt_big, kz_big, kg_big
        ], dim=0)
        return result_tensor
