import torch
import torch.nn as nn
from Class_cal_SF import cal_SF_mulband
from Class_p_lc import cal_P_lc
from Class_p_lt import cal_P_lt

class SingleBRF_nn_mulband(nn.Module):
    '''
    这个函数与原有函数的区别在于，这个函数是多波段的，而原有函数是单波段的
    '''
    def __init__(self,size_x,size_y,g,bL,bandnum):
        super(SingleBRF_nn_mulband, self).__init__()
        self.G=g
        self.size_x=size_x
        self.bandnum=bandnum
        self.SF_nn=cal_SF_mulband(size_x,size_y,bandnum)
        self.P_lc_nn=cal_P_lc(size_x,g,bL)
        self.P_lt_nn=cal_P_lt(size_x,g)

    def forward(self, input_tensor):
        # 生成所有结果组件（避免任何原地操作）
        components = []
        test_row=4
        test_col=69

        # ------ 第一部分计算结果 ------
        BRF_s = input_tensor[3] * input_tensor[14 + 2 * self.bandnum:14 + 3 * self.bandnum]

        # 计算中间变量
        part1 = input_tensor[[8, 9]]  # sza, saa
        part2 = input_tensor[14:14 + self.bandnum * 2]  # rl, tl
        SF = self.SF_nn(torch.cat([part1, part2], dim=0))

        P_lc = self.P_lc_nn(input_tensor[[10, 8, 11]])  # FAVD, sza, max_Ac
        P_lt = self.P_lt_nn(input_tensor[[10, 8, 12, 13]])  # FAVD, sza, max_At, mean_At

        # 计算BRF分量
        cos_sza = torch.cos(input_tensor[8])
        BRF_lc = (input_tensor[0] * input_tensor[10] * SF * P_lc) / (cos_sza)
        BRF_lt = (input_tensor[1] * input_tensor[10] * SF * P_lt) / (cos_sza)

        dim_len = len(BRF_lc.shape)
        if dim_len == 2:
            BRF_l = (BRF_lc+BRF_lt).unsqueeze(0)
        else:
            BRF_l = BRF_lc + BRF_lt
        BRF_single = BRF_l + BRF_s

        # 收集第一部分结果
        components.extend([BRF_s, BRF_l, BRF_single])

        # ------ 第二部分计算结果（使用big参数）------
        BRF_s_big = input_tensor[7] * input_tensor[14 + 2 * self.bandnum:14 + 3 * self.bandnum]

        BRF_lc_big = (input_tensor[4] * input_tensor[10] * SF * P_lc) / (cos_sza)
        BRF_lt_big = (input_tensor[5] * input_tensor[10] * SF * P_lt) / (cos_sza)

        dim_len = len(BRF_lc_big.shape)
        if dim_len == 2:
            BRF_l_big = (BRF_lc_big + BRF_lt_big).unsqueeze(0)
        else:
            BRF_l_big = BRF_lc_big + BRF_lt_big
        BRF_single_big = BRF_l_big + BRF_s_big

        # 收集第二部分结果
        components.extend([BRF_s_big, BRF_l_big, BRF_single_big])

        # 合并所有结果（避免任何切片赋值）
        return BRF_single_big
