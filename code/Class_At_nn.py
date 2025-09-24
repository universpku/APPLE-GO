import torch
import torch.nn as nn

class At_nn_tensor(nn.Module):
    '''
    计算At的神经网络
    '''
    def __init__(self,block_size_x,block_size_y,g=0.5):
        '''
        初始化函数，常规情况下block_size_x=block_size_y
        :param block_size_x: x方向的大像元包含的小像元数
        :param block_size_y: y方向的大像元包含的小像元数
        '''
        super(At_nn_tensor, self).__init__()
        self.block_size_x=block_size_x
        self.block_size_y=block_size_y
        self.G = g


    def old_forward(self,input_tensor):
    # def forward(self, input_tensor):
        '''
        计算At的函数
        :param input_tenser: 需要是一个[6,n,n]的张量，第一个通道是CHM，第二个通道是PATH, 第三个通道是FAVD, 第四个通道是Edge, 第五个通道是TH, 第六个通道是Max_H
        :return:[2,n,m]的矩阵，diyi是GAP_Vertical_At
        '''
        num_block_x = int(input_tensor.shape[1] / self.block_size_x)
        num_block_y = int(input_tensor.shape[2] / self.block_size_y)
        result_output = torch.zeros([4, num_block_x, num_block_y])
        input_tensor[input_tensor < 0] = 0
        result_tensor = torch.zeros([4, input_tensor.shape[1], input_tensor.shape[2]])
        result_tensor[1:4, :, :] = input_tensor[0, :, :]  - input_tensor[4, :, :]
        result_tensor[1, :, :] = result_tensor[1, :, :] * input_tensor[2, :, :] * self.G * (-1)
        result_tensor[1, :, :] = torch.exp(result_tensor[1, :, :])

        input_tensor[1:4, :, :][input_tensor[1:4, :, :] > 0] = 1
        result_tensor[0, :, :] = input_tensor[1, :, :] + input_tensor[3, :, :]
        result_tensor[0, :, :][result_tensor[0, :, :] > 0] = 1
        result_tensor[0, :, :] = input_tensor[0, :, :] * result_tensor[0, :, :]
        result_tensor[0, :, :][result_tensor[0, :, :] > 0] = 1
        result_tensor[1:4, :, :] = result_tensor[1:4, :, :] * result_tensor[0, :, :]

        result_output[0:2, :, :] = result_tensor[0:2, :, :].reshape(2, num_block_x, self.block_size_x, num_block_y, self.block_size_y).sum(
            dim=(2, 4))
        result_output[2, :, :] = result_tensor[2, :, :].reshape(1, num_block_x, self.block_size_x, num_block_y, self.block_size_y).max(
            dim=2)[0].max(dim=3)[0]
        result_output[3, :, :] = result_tensor[3, :, :].reshape(1, num_block_x, self.block_size_x, num_block_y, self.block_size_y).sum(
            dim=(2, 4))
        result_output[1, :, :] = result_output[1, :, :] / (result_output[0, :, :]+1e-8)
        result_output[3, :, :] = result_output[3, :, :] / (result_output[0, :, :]+1e-8)
        return result_output

    def forward(self, input_tensor):
        '''
        计算At的函数
        :param input_tensor: [6, H, W] 张量，通道顺序：CHM, PATH, FAVD, Edge, TH, Max_H
        :return: [4, num_block_x, num_block_y] 矩阵，通道：计数, GAP_Vertical_At, Max_At, 累计值
        '''
        # 消除所有原地操作
        num_block_x = input_tensor.shape[1] // self.block_size_x
        num_block_y = input_tensor.shape[2] // self.block_size_y

        # 非原地处理输入张量
        input_clean = torch.where(input_tensor < 0,
                                  torch.tensor(0.0, device=input_tensor.device),
                                  input_tensor)

        # 重构通道计算逻辑
        # 通道1-3基础计算
        base_diff_0 = input_clean[0] - input_clean[4]
        base_diff = torch.where(base_diff_0 < 0,
                                  torch.tensor(0.0, device=input_tensor.device),
                                  base_diff_0)

        deleteNumber = base_diff[base_diff > 0]
        if deleteNumber.numel() > 0:
            min_positive = torch.min(deleteNumber)
            # print("大于0的最小值:", min_positive.item())  # 输出标量值
        else:
            min_positive = 0
            # print("Tensor中没有大于0的元素")
        min_positive = 0
        base_diff = base_diff - min_positive  # 减去最小正数以避免负值

        ch1 = base_diff * input_clean[2] * self.G * (-1)
        ch1 = torch.exp(ch1)
        ch2 = base_diff.clone()
        ch3 = base_diff.clone()

        # 重构通道0计算逻辑
        # 原：input_tensor[1:4][input_tensor[1:4]>0] = 1 → 非原地版本
        mask_1to4 = (input_clean[1:4] > 0)
        modified_1to4 = torch.where(mask_1to4,
                                    torch.tensor(1.0, device=input_tensor.device),
                                    input_clean[1:4])

        # 通道0分步计算
        ch0_initial = modified_1to4[0] + modified_1to4[2]  # PATH + Edge
        ch0_mask = (ch0_initial > 0)
        ch0 = torch.where(ch0_mask,
                          torch.tensor(1.0, device=input_tensor.device),
                          torch.tensor(0.0, device=input_tensor.device))
        ch0 = input_clean[0] * ch0
        ch0 = torch.where(ch0 > 0,
                          torch.tensor(1.0, device=input_tensor.device),
                          torch.tensor(0.0, device=input_tensor.device))

        # 应用通道0的掩码
        ch1_masked = ch1 * ch0
        ch2_masked = ch2 * ch0
        ch3_masked = ch3 * ch0

        # 分块聚合函数
        def block_aggregate(tensor, mode='sum'):
            view_shape = (1, num_block_x, self.block_size_x, num_block_y, self.block_size_y)
            reshaped = tensor.view(view_shape)
            if mode == 'sum':
                return reshaped.sum(dim=(2, 4))
            elif mode == 'max':
                return reshaped.amax(dim=(2, 4))
            return reshaped

        # 并行计算所有通道聚合
        ch0_agg = block_aggregate(ch0, 'sum')  # 通道0聚合
        ch1_agg = block_aggregate(ch1_masked, 'sum')  # 通道1聚合
        ch2_agg = block_aggregate(ch2_masked, 'max')  # 通道2聚合
        ch3_agg = block_aggregate(ch3_masked, 'sum')  # 通道3聚合

        # 构建最终输出
        result_output = torch.stack([
            ch0_agg,  # 通道0: 计数
            ch1_agg / (ch0_agg + 1e-8),  # 通道1: GAP_Vertical_At
            ch2_agg,  # 通道2: Max_At
            ch3_agg / (ch0_agg + 1e-8)  # 通道3: 累计值
        ], dim=0)

        return result_output

    def forward_save(self, input_tensor):
        '''
        计算At的函数
        :param input_tensor: [6, H, W] 张量，通道顺序：CHM, PATH, FAVD, Edge, TH, Max_H
        :return: [4, num_block_x, num_block_y] 矩阵，通道：计数, GAP_Vertical_At, Max_At, 累计值
        '''
        # 消除所有原地操作
        num_block_x = input_tensor.shape[1] // self.block_size_x
        num_block_y = input_tensor.shape[2] // self.block_size_y

        # 非原地处理输入张量
        input_clean = torch.where(input_tensor < 0,
                                  torch.tensor(0.0, device=input_tensor.device),
                                  input_tensor)

        # 重构通道计算逻辑
        # 通道1-3基础计算
        base_diff = input_clean[0] - input_clean[4]
        ch1 = base_diff * input_clean[2] * self.G * (-1)
        ch1 = torch.exp(ch1)
        ch2 = base_diff.clone()
        ch3 = base_diff.clone()

        # 重构通道0计算逻辑
        # 原：input_tensor[1:4][input_tensor[1:4]>0] = 1 → 非原地版本
        mask_1to4 = (input_clean[1:4] > 0)
        modified_1to4 = torch.where(mask_1to4,
                                    torch.tensor(1.0, device=input_tensor.device),
                                    input_clean[1:4])

        # 通道0分步计算
        ch0_initial = modified_1to4[0] + modified_1to4[2]  # PATH + Edge
        ch0_mask = (ch0_initial > 0)
        ch0 = torch.where(ch0_mask,
                          torch.tensor(1.0, device=input_tensor.device),
                          torch.tensor(0.0, device=input_tensor.device))
        ch0 = input_clean[0] * ch0
        ch0 = torch.where(ch0 > 0,
                          torch.tensor(1.0, device=input_tensor.device),
                          torch.tensor(0.0, device=input_tensor.device))

        # 应用通道0的掩码
        ch1_masked = ch1 * ch0
        ch2_masked = ch2 * ch0
        ch3_masked = ch3 * ch0

        # 分块聚合函数
        def block_aggregate(tensor, mode='sum'):
            view_shape = (1, num_block_x, self.block_size_x, num_block_y, self.block_size_y)
            reshaped = tensor.view(view_shape)
            if mode == 'sum':
                return reshaped.sum(dim=(2, 4))
            elif mode == 'max':
                return reshaped.amax(dim=(2, 4))
            return reshaped

        # 并行计算所有通道聚合
        ch0_agg = block_aggregate(ch0, 'sum')  # 通道0聚合
        ch1_agg = block_aggregate(ch1_masked, 'sum')  # 通道1聚合
        ch2_agg = block_aggregate(ch2_masked, 'max')  # 通道2聚合
        ch3_agg = block_aggregate(ch3_masked, 'sum')  # 通道3聚合

        # 构建最终输出
        result_output = torch.stack([
            ch0_agg,  # 通道0: 计数
            ch1_agg / (ch0_agg + 1e-8),  # 通道1: GAP_Vertical_At
            ch2_agg,  # 通道2: Max_At
            ch3_agg / (ch0_agg + 1e-8)  # 通道3: 累计值
        ], dim=0)

        return result_output