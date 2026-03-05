import torch
import torch.nn as nn

from Class_Ag_nn import Ag_nn_tensor
from Class_Az_nn import Az_nn_tensor
from Class_Ac_nn import Ac_nn_tensor
from Class_At_nn import At_nn_tensor
from Class_Ac_Ag_big import Ac_Ag_big_nn_tensor
from Class_cal_k import cal_k
from Class_cal_LAI import cal_LAI

class Single_k_nn_0303(nn.Module):
    '''
    from input cal 8 k
    use；Ag_nn,Az_nn,Ac_nn,At_nn,Ac_Ag_nn and et. al.
    no_use:cal_k(instand direct use all cal in cal_k)
    '''
    def __init__(self,block_size_x,block_size_y,g=0.5):
        super(Single_k_nn_0303, self).__init__()
        self.block_size_x=block_size_x
        self.block_size_y=block_size_y
        self.G=g
        self.Ag_nn=Ag_nn_tensor(block_size_x,block_size_y)
        self.Az_nn=Az_nn_tensor(block_size_x,block_size_y,g)
        self.Ac_nn=Ac_nn_tensor(block_size_x,block_size_y,g)
        self.At_nn=At_nn_tensor(block_size_x,block_size_y,g)
        self.Ac_Ag_nn=Ac_Ag_big_nn_tensor(block_size_x,block_size_y,g)
        self.Edge_nn=nn.Conv2d(1,1,kernel_size=(3,3),stride=1)
        self.Edge_nn.weight=torch.nn.Parameter(torch.tensor([[[[-1.,-1,-1],
                                        [-1,0,-1],
                                        [-1,-1,-1]]]]), requires_grad=False)
        self.Edge_nn.bias= torch.nn.Parameter(torch.tensor([8.0]), requires_grad=False)
        self.cal_k_nn=cal_k(block_size_x,block_size_y)
        self.LAI_nn=cal_LAI(block_size_x,block_size_y,g)



    def forward(self, input_tensor):
        '''

        :param input_tensor:
                0:CHM
                1:PATH1
                2:PATH2
                3:FAVD
                4:TH
                5:Max_H
                6:sza


        return:
        '''

        # 重构 input_tensor_small_Ac_At 的创建
        input_tensor_small_Ac_At = torch.stack([
            input_tensor[0],  # 原索引0
            input_tensor[1],  # 原索引1
            input_tensor[3],  # 原索引3
            input_tensor[7],  # 原索引7
            input_tensor[4],  # 原索引4
            input_tensor[5]  # 原索引5
        ], dim=0)

        # 分步构建 tensor_output
        outputs = []

        # Ag_nn 的输出 (索引0)
        ag_output = self.Ag_nn(input_tensor[0:2].clone())
        outputs.append(ag_output)  # 第0位

        # Az_nn 的输出 (索引1)
        az_output = self.Az_nn(input_tensor[[0, 1, 3]].clone())
        outputs.append(az_output)  # 第1位

        # Ac_nn 的输出 (索引2,3,10)
        ac_outputs = self.Ac_nn(input_tensor_small_Ac_At.clone())
        outputs.append(ac_outputs[0])  # 第2位
        outputs.append(ac_outputs[1])  # 第3位

        # 填充中间空白索引（4-9）
        outputs.extend([torch.zeros_like(ag_output) for _ in range(6)])  # 索引4-9

        outputs.append(ac_outputs[2])  # 索引10

        # At_nn 的输出 (索引4,11,12)
        at_outputs = self.At_nn(input_tensor_small_Ac_At.clone())
        outputs[4] = at_outputs[1]  # 替换索引4
        outputs.append(at_outputs[2])  # 索引11
        outputs.append(at_outputs[3])  # 索引12

        # Ac_Ag_nn 的输出 (索引5-6)
        ac_ag_outputs = self.Ac_Ag_nn(input_tensor[[0, 1, 3, 2]].clone())
        outputs[5] = ac_ag_outputs[0].unsqueeze(0)  # 索引5
        outputs[6] = ac_ag_outputs[1].unsqueeze(0)  # 索引6

        LAI_output = self.LAI_nn(input_tensor_small_Ac_At.clone())
        outputs[7] = LAI_output  # 索引7

        # 合并所有输出
        tensor_output = torch.cat(outputs, dim=0)

        return tensor_output
