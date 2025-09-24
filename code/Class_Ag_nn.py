import torch
import torch.nn as nn


class Ag_nn_tensor(nn.Module):
    '''
    计算Ag的神经网络
    '''
    def __init__(self,block_size_x,block_size_y):
        '''
        初始化函数，常规情况下block_size_x=block_size_y
        :param block_size_x:x方向的大像元包含的小像元数
        :param block_size_y:y方向的大像元包含的小像元数
        '''
        super(Ag_nn_tensor, self).__init__()
        self.block_size_x=block_size_x
        self.block_size_y=block_size_y

    def old_forward(self, input_tensor):
    # def forward(self, input_tensor):
        '''
        计算Ag的函数,CHM与Path均为0的像元为Ag
        :param input_tenser: 需要是一个[2,n,n]的张量，第一个通道是CHM，第二个通道是PATH
        :return:
        '''
        input_tensor[input_tensor<0]=0
        # 将两层的张量相加，得到一个[1,n,n]的张量
        result_tensor=input_tensor.sum(dim=0)
        # 将结果张量中值大于0的像元置为0，等于0的像元置为1
        result_tensor[result_tensor>0]=-1
        result_tensor+=1
        num_block_x=int(input_tensor.shape[1]/self.block_size_x)
        num_block_y=int(input_tensor.shape[2]/self.block_size_y)
        result_tensor=result_tensor.reshape(1,num_block_x,self.block_size_x,num_block_y,self.block_size_y).sum(dim=(2,4))
        # print(result_tensor)
        # exit(-11)
        return result_tensor

    def forward(self, input_tensor):
        '''
        计算Ag的函数，CHM与Path均为0的像元为Ag
        :param input_tensor: 需要是一个[2, H, W]的张量，第一个通道是CHM，第二个通道是PATH
        :return: 聚合后的结果张量 [1, num_block_x, num_block_y]
        '''
        # 替换所有原地操作为安全操作
        # 原：input_tensor[input_tensor < 0] = 0
        input_tensor = torch.where(input_tensor < 0, 0, input_tensor)  # 创建新张量代替原地修改

        # 计算非原地相加
        result_tensor = input_tensor.sum(dim=0, keepdim=True)  # 保持维度 [1, H, W]

        # 重构条件判断逻辑
        # 原：result_tensor[result_tensor > 0] = -1 → result_tensor += 1
        mask = (result_tensor > 0)  # 创建布尔掩码
        result_tensor = torch.where(mask,
                                    torch.tensor(-1.0, device=input_tensor.device),  # 符合条件设为-1
                                    result_tensor)  # 不符合保持原值
        result_tensor = result_tensor + 1  # 全部元素+1 (非原地操作)

        # 分块聚合
        num_block_x = input_tensor.shape[1] // self.block_size_x
        num_block_y = input_tensor.shape[2] // self.block_size_y

        # 重构reshape和sum操作
        result_tensor = result_tensor.view(
            1,
            num_block_x, self.block_size_x,
            num_block_y, self.block_size_y
        ).sum(dim=(2, 4))  # 同时求和两个维度

        return result_tensor