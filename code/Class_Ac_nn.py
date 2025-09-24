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

    def old_forward(self, input_tensor):
    # def forward(self, input_tensor):
        '''
        计算Ac的函数
        :param input_tenser: 需要是一个[6,n,n]的张量，0:第一个通道是CHM，1:第二个通道是PATH, 2:第三个通道是FAVD, 3:第四个通道是Edge, 4:第五个通道是TH, 5:第六个通道是Max_H
        :return:[2,n,m]的矩阵，第一个通道是count_Ac,第二个通道是GAP_Vertical_Ac
        '''
        num_block_x = int(input_tensor.shape[1] / self.block_size_x)
        num_block_y = int(input_tensor.shape[2] / self.block_size_y)
        result_output = torch.zeros([3, num_block_x, num_block_y])
        input_tensor[input_tensor < 0] = 0
        result_tensor = torch.zeros([3, input_tensor.shape[1], input_tensor.shape[2]])
        result_tensor[1, :, :] = input_tensor[0, :, :]-input_tensor[4,:,:]
        result_tensor[2, :, :] = input_tensor[0, :, :]-input_tensor[4,:,:]
        # print(result_tensor[1,0,14],input_tensor[0, 0, 14],input_tensor[4,0,14],input_tensor[5,0,14],(14974 * torch.exp(-2.037*input_tensor[0,0,14])))
        result_tensor[1, :, :] = result_tensor[1, :, :]*input_tensor[2,:,:]*self.G*(-1)
        result_tensor[1, :, :] = torch.exp(result_tensor[1, :, :])

        input_tensor[1:4,:,:][input_tensor[1:4,:,:]>0]=-1
        input_tensor[1:4, :, :] += 1
        result_tensor[0, :, :] = input_tensor[0, :, :] * input_tensor[1, :, :]*input_tensor[3, :, :]
        result_tensor[0, :, :][result_tensor[0, :, :] > 0] = 1
        result_tensor[1:3, :, :] = result_tensor[1:3, :, :] * result_tensor[0, :, :]



        result_output[0:2,:,:] = result_tensor[0:2,:,:].reshape(2, num_block_x, self.block_size_x, num_block_y, self.block_size_y).sum(dim=(2, 4))
        result_output[1,:,:]=result_output[1,:,:]/(result_output[0,:,:]+1e-8)
        result_output[2, :, :] = result_tensor[2, :, :].reshape(1, num_block_x, self.block_size_x, num_block_y, self.block_size_y).max(dim=2)[
            0].max(dim=3)[0]
        # print("result_output[:,1,2]",result_output[:,1,2])
        # exit(-179)

        return result_output

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
            # print("大于0的最小值:", min_positive.item())  # 输出标量值
        else:
            min_positive = 0
            # print("Tensor中没有大于0的元素")
        # min_positive=0

        ch1 = (ch1_base-min_positive) * input_clean[2] * self.G * (-1)
        ch1 = torch.exp(ch1)

        # 通道2计算
        ch2_base = input_clean[0] - input_clean[4]

        # 通道0计算逻辑重构
        # 原：input_tensor[1:4][input_tensor[1:4]>0] = -1 → input_tensor[1:4] += 1
        mask_1to4 = (input_clean[1:4] > 0)
        modified_1to4 = torch.where(mask_1to4,
                                    torch.tensor(-1.0, device=input_tensor.device),
                                    input_clean[1:4])
        modified_1to4 += 1  # 此时操作不会影响原始输入

        # 计算通道0
        # ch0 = input_clean[0] * modified_1to4[0] * modified_1to4[2]
        # ch0 = torch.where(ch0 > 0,
        #                   torch.tensor(1.0, device=input_tensor.device),
        #                   torch.tensor(0.0, device=input_tensor.device))

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

    def forward_save(self, input_tensor):
        '''
        0630保存的原版forward
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
        ch1_base = input_clean[0] - input_clean[4]
        ch1 = ch1_base * input_clean[2] * self.G * (-1)
        ch1 = torch.exp(ch1)

        # 通道2计算
        ch2_base = input_clean[0] - input_clean[4]

        # 通道0计算逻辑重构
        # 原：input_tensor[1:4][input_tensor[1:4]>0] = -1 → input_tensor[1:4] += 1
        mask_1to4 = (input_clean[1:4] > 0)
        modified_1to4 = torch.where(mask_1to4,
                                    torch.tensor(-1.0, device=input_tensor.device),
                                    input_clean[1:4])
        modified_1to4 += 1  # 此时操作不会影响原始输入

        # 计算通道0
        ch0 = input_clean[0] * modified_1to4[0] * modified_1to4[2]
        ch0 = torch.where(ch0 > 0,
                          torch.tensor(1.0, device=input_tensor.device),
                          torch.tensor(0.0, device=input_tensor.device))

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

    def forward_test52(self, input_tensor,path2_bottom):
        '''
        计算Ac的函数,用于5.2测试
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
        ch1_base = input_clean[0] - input_clean[4]
        ch1 = ch1_base * input_clean[2] * self.G * (-1)
        ch1 = torch.exp(ch1)

        # 通道2计算
        ch2_base = input_clean[0] - input_clean[4]

        # 通道0计算逻辑重构
        # 原：input_tensor[1:4][input_tensor[1:4]>0] = -1 → input_tensor[1:4] += 1
        mask_1to4 = (input_clean[1:4] > 0)
        modified_1to4 = torch.where(mask_1to4,
                                    torch.tensor(-1.0, device=input_tensor.device),
                                    input_clean[1:4])
        modified_1to4 += 1  # 此时操作不会影响原始输入

        # 计算通道0
        ch0 = input_clean[0] * modified_1to4[0] * modified_1to4[2]
        ch0 = torch.where(ch0 > 0,
                          torch.tensor(1.0, device=input_tensor.device),
                          torch.tensor(0.0, device=input_tensor.device))

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

        # chm_th= input_clean[0]-input_clean[4]
        # 并行计算各通道聚合
        ch0_agg = block_aggregate(result_tensor[0], 'sum')  # count_Ac
        ch1_agg = block_aggregate(result_tensor[1], 'sum')  # 分子
        ch2_agg = block_aggregate(result_tensor[2], 'max')  # Max_Ac
        ch3_agg = block_aggregate(ch2_base, 'sum')  # Sum_Ac0
        ch4_agg = block_aggregate(path2_bottom, 'sum')  # Sum_Ac_sita

        # 构建最终输出
        result_output = torch.stack([
            ch0_agg,  # 通道0
            ch1_agg / (ch0_agg + 1e-8),  # 通道1（GAP_Vertical_Ac）
            ch2_agg,  # 通道2
            ch4_agg / (ch3_agg + 1e-8)
        ], dim=0)

        return result_output