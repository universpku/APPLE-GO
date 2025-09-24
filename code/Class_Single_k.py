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
        # cal_edge
        # tensor_CHM=torch.Tensor(input_tensor[0,:,:].clone()).unsqueeze(0).unsqueeze(0)
        # tensor_CHM[tensor_CHM>0]=1
        # tensor_edge=torch.zeros([1,input_tensor.shape[1],input_tensor.shape[2]])
        # tensor_edge[0,1:-1,1:-1]=self.Edge_nn(tensor_CHM)
        # tensor_edge[tensor_edge<=1]=0
        # tensor_edge[tensor_edge>1]=1
        # tensor_edge[tensor_CHM[0]==0]=0
        # tensor_CHM[0,0,:,:][input_tensor[0,:,:]>0]=2
        # tensor_CHM[0,:,:][tensor_edge==1]=1
        #
        num_block_x = int(input_tensor.shape[1] / self.block_size_x)
        num_block_y = int(input_tensor.shape[2] / self.block_size_y)
        # Count_edge = tensor_edge.reshape(1, num_block_x, self.block_size_x, num_block_y, self.block_size_y).sum(
        #     dim=(2, 4))
        # Gap_count = (2-tensor_CHM).reshape(1, num_block_x, self.block_size_x, num_block_y, self.block_size_y).sum(
        #     dim=(2, 4))/2

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
        # ag_output_old = self.Ag_nn.old_forward(input_tensor[0:2].clone())
        # # 比较ag_output和ag_output_old
        # compare=ag_output-ag_output_old
        # print("ag_output",ag_output)
        # print("ag_output_old",ag_output_old)
        # print("compare",compare)


        # Az_nn 的输出 (索引1)
        az_output = self.Az_nn(input_tensor[[0, 1, 3]].clone())
        outputs.append(az_output)  # 第1位
        # az_output_old = self.Az_nn.old_forward(input_tensor[[0, 1, 3]].clone())
        # # 比较az_output和az_output_old
        # compare=az_output-az_output_old
        # print("az_output",az_output)
        # print("az_output_old",az_output_old)
        # print("compare",compare)

        # Ac_nn 的输出 (索引2,3,10)
        ac_outputs = self.Ac_nn(input_tensor_small_Ac_At.clone())
        outputs.append(ac_outputs[0])  # 第2位
        outputs.append(ac_outputs[1])  # 第3位
        # ac_outputs_old = self.Ac_nn.old_forward(input_tensor_small_Ac_At.clone())
        # # 比较ac_outputs和ac_outputs_old
        # compare=ac_outputs[2]-ac_outputs_old[2]
        # print("ac_outputs[0]",ac_outputs[1])
        # print("ac_outputs_old[0]",ac_outputs_old[0])
        # print("compare",compare)

        # 填充中间空白索引（4-9）
        outputs.extend([torch.zeros_like(ag_output) for _ in range(6)])  # 索引4-9

        outputs.append(ac_outputs[2])  # 索引10

        # At_nn 的输出 (索引4,11,12)
        at_outputs = self.At_nn(input_tensor_small_Ac_At.clone())
        outputs[4] = at_outputs[1]  # 替换索引4
        outputs.append(at_outputs[2])  # 索引11
        outputs.append(at_outputs[3])  # 索引12
        # at_outputs_old = self.At_nn.old_forward(input_tensor_small_Ac_At.clone())
        # # 比较at_outputs和at_outputs_old
        # compare=at_outputs[3]-at_outputs_old[3]
        # print("at_outputs[1]",at_outputs[1])
        # print("at_outputs_old[1]",at_outputs_old[1])
        # print("compare",compare)

        # Ac_Ag_nn 的输出 (索引5-6)
        ac_ag_outputs = self.Ac_Ag_nn(input_tensor[[0, 1, 3, 2]].clone())
        outputs[5] = ac_ag_outputs[0].unsqueeze(0)  # 索引5
        outputs[6] = ac_ag_outputs[1].unsqueeze(0)  # 索引6
        # ac_ag_outputs_old = self.Ac_Ag_nn.old_forward(input_tensor[[0, 1, 3, 2]].clone())
        # # 比较ac_ag_outputs和ac_ag_outputs_old
        # compare=ac_ag_outputs[0]-ac_ag_outputs_old[0]
        # print("ac_ag_outputs[0]",ac_ag_outputs[0])
        # print("ac_ag_outputs_old[0]",ac_ag_outputs_old[0])
        # print("compare",compare)

        LAI_output = self.LAI_nn(input_tensor_small_Ac_At.clone())
        outputs[7] = LAI_output  # 索引7
        # LAI_output_old = self.LAI_nn.old_forward(input_tensor_small_Ac_At.clone())
        # # 比较LAI_output和LAI_output_old
        # compare=LAI_output-LAI_output_old
        # print("LAI_output",LAI_output)
        # print("LAI_output_old",LAI_output_old)
        # print("compare",compare)

        # 验证各模块输出形状
        # print(ag_output.shape)  # 应输出 (H,W)
        # print(az_output.shape)  # 应输出 (H,W)
        # print(ac_outputs.shape)  # 应输出 (3,H,W)
        # print(at_outputs.shape)  # 应输出 (>=4,H,W)
        # print(ac_ag_outputs.shape)  # 应输出 (2,H,W)
        # print(LAI_output.shape)  # 应输出 (1,H,W)

        # 合并所有输出
        tensor_output = torch.cat(outputs, dim=0)
        # print("tensor_output.shape", tensor_output.shape)

        return tensor_output
        # return result_tensor

    def forward_test52(self, input_tensor,PATH2_bottom):
        '''
        该函数的目的在于测试路径长度使用的方式修改等过程
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
        # cal_edge
        # tensor_CHM=torch.Tensor(input_tensor[0,:,:].clone()).unsqueeze(0).unsqueeze(0)
        # tensor_CHM[tensor_CHM>0]=1
        # tensor_edge=torch.zeros([1,input_tensor.shape[1],input_tensor.shape[2]])
        # tensor_edge[0,1:-1,1:-1]=self.Edge_nn(tensor_CHM)
        # tensor_edge[tensor_edge<=1]=0
        # tensor_edge[tensor_edge>1]=1
        # tensor_edge[tensor_CHM[0]==0]=0
        # tensor_CHM[0,0,:,:][input_tensor[0,:,:]>0]=2
        # tensor_CHM[0,:,:][tensor_edge==1]=1
        #
        num_block_x = int(input_tensor.shape[1] / self.block_size_x)
        num_block_y = int(input_tensor.shape[2] / self.block_size_y)
        # Count_edge = tensor_edge.reshape(1, num_block_x, self.block_size_x, num_block_y, self.block_size_y).sum(
        #     dim=(2, 4))
        # Gap_count = (2-tensor_CHM).reshape(1, num_block_x, self.block_size_x, num_block_y, self.block_size_y).sum(
        #     dim=(2, 4))/2

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
        # ag_output_old = self.Ag_nn.old_forward(input_tensor[0:2].clone())
        # # 比较ag_output和ag_output_old
        # compare=ag_output-ag_output_old
        # print("ag_output",ag_output)
        # print("ag_output_old",ag_output_old)
        # print("compare",compare)


        # Az_nn 的输出 (索引1)
        az_output = self.Az_nn(input_tensor[[0, 1, 3]].clone())
        outputs.append(az_output)  # 第1位
        # az_output_old = self.Az_nn.old_forward(input_tensor[[0, 1, 3]].clone())
        # # 比较az_output和az_output_old
        # compare=az_output-az_output_old
        # print("az_output",az_output)
        # print("az_output_old",az_output_old)
        # print("compare",compare)

        # Ac_nn 的输出 (索引2,3,10)
        ac_outputs = self.Ac_nn.forward_test52(input_tensor_small_Ac_At.clone(),PATH2_bottom)
        outputs.append(ac_outputs[0])  # 第2位
        outputs.append(ac_outputs[1])  # 第3位
        # ac_outputs_old = self.Ac_nn.old_forward(input_tensor_small_Ac_At.clone())
        # # 比较ac_outputs和ac_outputs_old
        # compare=ac_outputs[2]-ac_outputs_old[2]
        # print("ac_outputs[0]",ac_outputs[1])
        # print("ac_outputs_old[0]",ac_outputs_old[0])
        # print("compare",compare)

        # 填充中间空白索引（4-9）
        outputs.extend([torch.zeros_like(ag_output) for _ in range(6)])  # 索引4-9
        outputs[9] = ac_outputs[3]

        outputs.append(ac_outputs[2])  # 索引10

        # At_nn 的输出 (索引4,11,12)
        at_outputs = self.At_nn(input_tensor_small_Ac_At.clone())
        outputs[4] = at_outputs[1]  # 替换索引4
        outputs.append(at_outputs[2])  # 索引11
        outputs.append(at_outputs[3])  # 索引12
        # at_outputs_old = self.At_nn.old_forward(input_tensor_small_Ac_At.clone())
        # # 比较at_outputs和at_outputs_old
        # compare=at_outputs[3]-at_outputs_old[3]
        # print("at_outputs[1]",at_outputs[1])
        # print("at_outputs_old[1]",at_outputs_old[1])
        # print("compare",compare)

        # Ac_Ag_nn 的输出 (索引5-6)
        ac_ag_outputs = self.Ac_Ag_nn(input_tensor[[0, 1, 3, 2]].clone())
        outputs[5] = ac_ag_outputs[0].unsqueeze(0)  # 索引5
        outputs[6] = ac_ag_outputs[1].unsqueeze(0)  # 索引6
        # ac_ag_outputs_old = self.Ac_Ag_nn.old_forward(input_tensor[[0, 1, 3, 2]].clone())
        # # 比较ac_ag_outputs和ac_ag_outputs_old
        # compare=ac_ag_outputs[0]-ac_ag_outputs_old[0]
        # print("ac_ag_outputs[0]",ac_ag_outputs[0])
        # print("ac_ag_outputs_old[0]",ac_ag_outputs_old[0])
        # print("compare",compare)

        LAI_output = self.LAI_nn(input_tensor_small_Ac_At.clone())
        outputs[7] = LAI_output  # 索引7
        # LAI_output_old = self.LAI_nn.old_forward(input_tensor_small_Ac_At.clone())
        # # 比较LAI_output和LAI_output_old
        # compare=LAI_output-LAI_output_old
        # print("LAI_output",LAI_output)
        # print("LAI_output_old",LAI_output_old)
        # print("compare",compare)

        # 验证各模块输出形状
        # print(ag_output.shape)  # 应输出 (H,W)
        # print(az_output.shape)  # 应输出 (H,W)
        # print(ac_outputs.shape)  # 应输出 (3,H,W)
        # print(at_outputs.shape)  # 应输出 (>=4,H,W)
        # print(ac_ag_outputs.shape)  # 应输出 (2,H,W)
        # print(LAI_output.shape)  # 应输出 (1,H,W)

        # 合并所有输出
        tensor_output = torch.cat(outputs, dim=0)
        # print("tensor_output.shape", tensor_output.shape)

        return tensor_output
        # return result_tensor