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
        # print("in SingleBRF_nn_mulband __init__",bL)

    def old_forward(self,input_tensor):
        '''

        :param input_tensor:
            0:kc
            1:kt
            2:kz
            3:kg
            4:big kc
            5:big kt
            6:big kz
            7:big kg
            8:sza
            9:saa
            10:FAVD
            11:max_Ac
            12:max_At
            13:mean_At
            14-14+bandnum:rl
            14+bandnum-14+2*bandnum:tl
            14+2*bandnum:14+3*bandnum:rs
        :return:
        '''
        # print("SingleBRF_nn_mulband")
        # print(input_tensor[:,1,2])

        result_tensor = torch.zeros([6*self.bandnum, input_tensor.shape[1], input_tensor.shape[2]])
        BRF_s=input_tensor[3,:,:]*input_tensor[14+2*self.bandnum:14+3*self.bandnum,:,:]
        #0：sza.1:saa,2:rl,3:tl
        part1=input_tensor[[8,9],:,:]
        part2=input_tensor[14:14+self.bandnum*2,:,:]
        SF=self.SF_nn(torch.cat([part1,part2],dim=0))
        #0:FAVD,1:sza，2：max_Ac
        P_lc=self.P_lc_nn(input_tensor[[10,8,11],:,:])
        #0:FAVD,1:sza，2：max_At,3:mean_At
        P_lt=self.P_lt_nn(input_tensor[[10,8,12,13],:,:])

        # print("P_lc", P_lc[1, 2])
        # print("P_lt", P_lt[1, 2])
        # print("SF", SF[1, 2])

        BRF_lc=input_tensor[0,:,:]*input_tensor[10,:,:]*SF*P_lc/torch.cos(input_tensor[8,:,:])/torch.cos(input_tensor[8,:,:])
        BRF_lt=input_tensor[1,:,:]*input_tensor[10,:,:]*SF*P_lt/torch.cos(input_tensor[8,:,:])/torch.cos(input_tensor[8,:,:])/torch.cos(input_tensor[8,:,:])
        BRF_l=BRF_lc+BRF_lt
        # print("BRF_l",BRF_l[1,2])
        # print("BRF_lc",BRF_lc[1,2])
        # print("BRF_lt",P_lc[1,2])
        # print("BRF_lt", P_lc[1, 3])
        # exit(-698)
        BRF_single=BRF_l+BRF_s

        result_tensor[0:self.bandnum,:,:]=BRF_s
        result_tensor[self.bandnum*1:self.bandnum*2,:,:]=BRF_l
        result_tensor[self.bandnum*2:self.bandnum*3,:,:]=BRF_single
        BRF_s = input_tensor[7, :, :] * input_tensor[14+2*self.bandnum:14+3*self.bandnum, :, :]
        BRF_lc = input_tensor[4, :, :] * input_tensor[10, :, :] * SF * P_lc / torch.cos(
            input_tensor[8, :, :]) / torch.cos(input_tensor[8, :, :])
        BRF_lt = input_tensor[5, :, :] * input_tensor[10, :, :] * SF * P_lt / torch.cos(
            input_tensor[8, :, :]) / torch.cos(input_tensor[8, :, :]) / torch.cos(input_tensor[8, :, :])
        BRF_l = BRF_lc + BRF_lt
        BRF_single = BRF_l + BRF_s
        result_tensor[self.bandnum*3:self.bandnum*4, :, :] = BRF_s
        result_tensor[self.bandnum*4:self.bandnum*5, :, :] = BRF_l
        result_tensor[self.bandnum*5:self.bandnum*6, :, :] = BRF_single



        return result_tensor

    def forward(self, input_tensor):
        # 生成所有结果组件（避免任何原地操作）
        components = []
        test_row=0
        test_col=0

        # ------ 第一部分计算结果 ------
        BRF_s = input_tensor[3] * input_tensor[14 + 2 * self.bandnum:14 + 3 * self.bandnum]

        # 计算中间变量
        part1 = input_tensor[[8, 9]]  # sza, saa
        part2 = input_tensor[14:14 + self.bandnum * 2]  # rl, tl
        SF = self.SF_nn(torch.cat([part1, part2], dim=0))
        # print("SF")
        # print(SF[:,test_row,test_col])


        P_lc = self.P_lc_nn(input_tensor[[10, 8, 11]])  # FAVD, sza, max_Ac
        P_lt = self.P_lt_nn(input_tensor[[10, 8, 12, 13]])  # FAVD, sza, max_At, mean_At
        # print("P_lc")
        # print(P_lc[test_row, test_col])
        # print("P_lt")
        # print(P_lt[test_row, test_col])
        # print(input_tensor[[4,5,10,11,12],test_row,test_col])
        # print("small kc kt kz kg",input_tensor[0:4, test_row, test_col])
        # print("kc kt kz kg",input_tensor[4:8, test_row, test_col])

        # return P_lt.unsqueeze(0)
        # 计算BRF分量
        cos_sza = torch.cos(input_tensor[8])
        # P_lc=1
        # SF = 3.14*SF
        # P_lt=1
        # SF=input_tensor[14:14 + self.bandnum * 1]
        # BRF_lc = (input_tensor[0] * input_tensor[10] * SF * P_lc) / (cos_sza  **  2)
        # BRF_lt = (input_tensor[1] * input_tensor[10] * SF * P_lt) / (cos_sza  **  3)
        BRF_lc = (input_tensor[0] * input_tensor[10] * SF * P_lc) / (cos_sza)
        BRF_lt = (input_tensor[1] * input_tensor[10] * SF * P_lt) / (cos_sza)

        dim_len = len(BRF_lc.shape)
        if dim_len == 2:
            BRF_l = (BRF_lc+BRF_lt).unsqueeze(0)
        else:
            BRF_l = BRF_lc + BRF_lt
        BRF_single = BRF_l + BRF_s

        # return BRF_l

        # 收集第一部分结果
        components.extend([BRF_s, BRF_l, BRF_single])

        # ------ 第二部分计算结果（使用big参数）------
        BRF_s_big = input_tensor[7] * input_tensor[14 + 2 * self.bandnum:14 + 3 * self.bandnum]

        # BRF_lc_big = (input_tensor[4] * input_tensor[10] * SF * P_lc) / (cos_sza ** 2)
        # BRF_lt_big = (input_tensor[5] * input_tensor[10] * SF * P_lt) / (cos_sza ** 3)
        BRF_lc_big = (input_tensor[4] * input_tensor[10] * SF * P_lc) / (cos_sza)
        BRF_lt_big = (input_tensor[5] * input_tensor[10] * SF * P_lt) / (cos_sza)

        # BRF_lc_big = (input_tensor[4] * input_tensor[10] * SF * P_lc) / (cos_sza  **  2)* input_tensor[11]
        # BRF_lt_big = (input_tensor[5] * input_tensor[10] * SF * P_lt) / (cos_sza  **  3)* input_tensor[13]
        # print(test)

        # print("BRF_lc_big",BRF_lc_big.shape)
        # print(BRF_lc_big[:,test_row,test_col])
        # print("BRF_lt_big",BRF_lt_big.shape)
        # print(BRF_lt_big[:,test_row,test_col])
        # print("cos_sza",cos_sza[test_row,test_col])

        dim_len = len(BRF_lc_big.shape)
        if dim_len == 2:
            BRF_l_big = (BRF_lc_big + BRF_lt_big).unsqueeze(0)
        else:
            BRF_l_big = BRF_lc_big + BRF_lt_big
        BRF_single_big = BRF_l_big + BRF_s_big

        # 收集第二部分结果
        components.extend([BRF_s_big, BRF_l_big, BRF_single_big])
        # print("BRF_single_big",BRF_single_big.shape)
        # print(BRF_single_big[:,test_row,test_col])

        # for _ in components:
        #     print(_.shape)

        # 合并所有结果（避免任何切片赋值）
        return BRF_single_big
        # return torch.cat(components, dim=0)  # 最终形状 [6*bandnum, H, W]

    def forward_more(self, input_tensor):
        # 生成所有结果组件（避免任何原地操作）
        components = []
        test_row=1
        test_col=1

        # ------ 第一部分计算结果 ------
        BRF_s = input_tensor[3] * input_tensor[14 + 2 * self.bandnum:14 + 3 * self.bandnum]

        # 计算中间变量
        part1 = input_tensor[[8, 9]]  # sza, saa
        part2 = input_tensor[14:14 + self.bandnum * 2]  # rl, tl
        SF = self.SF_nn(torch.cat([part1, part2], dim=0))
        # print("SF")
        # print(SF[:,test_row,test_col])


        P_lc = self.P_lc_nn(input_tensor[[10, 8, 11]])  # FAVD, sza, max_Ac
        P_lt = self.P_lt_nn(input_tensor[[10, 8, 12, 13]])  # FAVD, sza, max_At, mean_At
        # print("P_lc")
        # print(P_lc[test_row, test_col])
        # print("P_lt")
        # print(P_lt[test_row, test_col])
        # print(input_tensor[[4,5,10,11,12],test_row,test_col])
        # print("kc kt kz kg",input_tensor[4:8, test_row, test_col])

        # return P_lt.unsqueeze(0)
        # 计算BRF分量
        cos_sza = torch.cos(input_tensor[8])
        # P_lc=1
        # SF = 3.14*SF
        # P_lt=1
        # SF=input_tensor[14:14 + self.bandnum * 1]
        BRF_lc = (input_tensor[0] * input_tensor[10] * SF * P_lc) / (cos_sza  **  2)
        BRF_lt = (input_tensor[1] * input_tensor[10] * SF * P_lt) / (cos_sza  **  3)

        dim_len = len(BRF_lc.shape)
        if dim_len == 2:
            BRF_l = (BRF_lc+BRF_lt).unsqueeze(0)
        else:
            BRF_l = BRF_lc + BRF_lt
        BRF_single = BRF_l + BRF_s

        # return BRF_l

        # 收集第一部分结果
        components.extend([BRF_s, BRF_l, BRF_single])

        # ------ 第二部分计算结果（使用big参数）------
        BRF_s_big = input_tensor[7] * input_tensor[14 + 2 * self.bandnum:14 + 3 * self.bandnum]

        # BRF_lc_big = (input_tensor[4] * input_tensor[10] * SF * P_lc) / (cos_sza ** 2)
        # BRF_lt_big = (input_tensor[5] * input_tensor[10] * SF * P_lt) / (cos_sza ** 3)

        # BRF_lc_big = ( input_tensor[10] * SF * P_lc) / (cos_sza **  2)* input_tensor[11]
        # BRF_lt_big = ( input_tensor[10] * SF * P_lt) / (cos_sza **  3)* input_tensor[13]

        # BRF_lc_big = (input_tensor[10] * SF * P_lc) / (cos_sza ** 2)
        # BRF_lt_big = (input_tensor[10] * SF * P_lt) / (cos_sza ** 3)

        BRF_lc_big = (input_tensor[10] * SF * P_lc) / (cos_sza)
        BRF_lt_big = (input_tensor[10] * SF * P_lt) / (cos_sza)
        # print(test)

        # print("BRF_lc_big",BRF_lc_big.shape)
        # print(BRF_lc_big[:,test_row,test_col])
        # print("BRF_lt_big",BRF_lt_big.shape)
        # print(BRF_lt_big[:,test_row,test_col])
        # print("cos_sza",cos_sza[test_row,test_col])

        dim_len = len(BRF_lc_big.shape)
        if dim_len == 2:
            BRF_l_big = (BRF_lc_big + BRF_lt_big).unsqueeze(0)
        else:
            BRF_l_big = BRF_lc_big + BRF_lt_big
        BRF_single_big = BRF_l_big + BRF_s_big

        # 收集第二部分结果
        components.extend([BRF_s_big, BRF_l_big, BRF_single_big])
        # print("BRF_single_big",BRF_single_big.shape)
        # print(BRF_single_big[:,test_row,test_col])

        # for _ in components:
        #     print(_.shape)

        # 合并所有结果（避免任何切片赋值）
        return BRF_lc_big,BRF_lt_big
        # return torch.cat(components, dim=0)  # 最终形状 [6*bandnum, H, W]

    def forward_more_test52(self, input_tensor,lrmean):
        # 生成所有结果组件（避免任何原地操作）
        components = []
        test_row = 1
        test_col = 1

        # ------ 第一部分计算结果 ------
        BRF_s = input_tensor[3] * input_tensor[14 + 2 * self.bandnum:14 + 3 * self.bandnum]

        # 计算中间变量
        part1 = input_tensor[[8, 9]]  # sza, saa
        part2 = input_tensor[14:14 + self.bandnum * 2]  # rl, tl
        SF = self.SF_nn(torch.cat([part1, part2], dim=0))
        # print("SF")
        # print(SF[:, test_row, test_col])

        # P_lc = self.P_lc_nn(input_tensor[[10, 8, 11]])  # FAVD, sza, max_Ac
        P_lc = self.P_lc_nn.forward_test52(input_tensor[[10, 8, 11]],lrmean)  # FAVD, sza, max_Ac
        P_lt = self.P_lt_nn(input_tensor[[10, 8, 12, 13]])  # FAVD, sza, max_At, mean_At
        print("P_lc")
        print(P_lc[test_row, test_col])
        # print("P_lt")
        # print(P_lt[test_row, test_col])
        # print(input_tensor[[4, 5, 10, 11, 12], test_row, test_col])
        # print("kc kt kz kg", input_tensor[4:8, test_row, test_col])

        # return P_lt.unsqueeze(0)
        # 计算BRF分量
        cos_sza = torch.cos(input_tensor[8])
        # P_lc=1
        # SF = 3.14*SF
        # P_lt=1
        # SF=input_tensor[14:14 + self.bandnum * 1]
        BRF_lc = (input_tensor[0] * input_tensor[10] * SF * P_lc) / (cos_sza ** 2)
        BRF_lt = (input_tensor[1] * input_tensor[10] * SF * P_lt) / (cos_sza ** 3)

        dim_len = len(BRF_lc.shape)
        if dim_len == 2:
            BRF_l = (BRF_lc + BRF_lt).unsqueeze(0)
        else:
            BRF_l = BRF_lc + BRF_lt
        BRF_single = BRF_l + BRF_s

        # return BRF_l

        # 收集第一部分结果
        components.extend([BRF_s, BRF_l, BRF_single])

        # ------ 第二部分计算结果（使用big参数）------
        BRF_s_big = input_tensor[7] * input_tensor[14 + 2 * self.bandnum:14 + 3 * self.bandnum]

        # BRF_lc_big = (input_tensor[4] * input_tensor[10] * SF * P_lc) / (cos_sza ** 2)
        # BRF_lt_big = (input_tensor[5] * input_tensor[10] * SF * P_lt) / (cos_sza ** 3)

        # BRF_lc_big = ( input_tensor[10] * SF * P_lc) / (cos_sza **  2)* input_tensor[11]
        # BRF_lt_big = ( input_tensor[10] * SF * P_lt) / (cos_sza **  3)* input_tensor[13]

        # BRF_lc_big = (input_tensor[10] * SF * P_lc) / (cos_sza ** 2)
        # BRF_lt_big = (input_tensor[10] * SF * P_lt) / (cos_sza ** 3)

        BRF_lc_big = (input_tensor[10] * SF * P_lc) / (cos_sza)
        BRF_lt_big = (input_tensor[10] * SF * P_lt) / (cos_sza)
        # print(test)

        # print("BRF_lc_big", BRF_lc_big.shape)
        # print(BRF_lc_big[:, test_row, test_col])
        # print("BRF_lt_big", BRF_lt_big.shape)
        # print(BRF_lt_big[:, test_row, test_col])
        # print("cos_sza", cos_sza[test_row, test_col])

        dim_len = len(BRF_lc_big.shape)
        if dim_len == 2:
            BRF_l_big = (BRF_lc_big + BRF_lt_big).unsqueeze(0)
        else:
            BRF_l_big = BRF_lc_big + BRF_lt_big
        BRF_single_big = BRF_l_big + BRF_s_big

        # 收集第二部分结果
        components.extend([BRF_s_big, BRF_l_big, BRF_single_big])
        print("BRF_single_big", BRF_single_big.shape)
        print(BRF_single_big[:, test_row, test_col])

        # for _ in components:
        #     print(_.shape)

        # 合并所有结果（避免任何切片赋值）
        return BRF_lc_big, BRF_lt_big
        # return torch.cat(components, dim=0)  # 最终形状 [6*bandnum, H, W]

    def forward_kc(self, input_tensor):
        # 生成所有结果组件（避免任何原地操作）
        components = []
        test_row=0
        test_col=0

        # ------ 第一部分计算结果 ------
        BRF_s = input_tensor[3] * input_tensor[14 + 2 * self.bandnum:14 + 3 * self.bandnum]

        # 计算中间变量
        part1 = input_tensor[[8, 9]]  # sza, saa
        part2 = input_tensor[14:14 + self.bandnum * 2]  # rl, tl
        SF = self.SF_nn(torch.cat([part1, part2], dim=0))
        # print("SF")
        # print(SF[:,test_row,test_col])


        P_lc = self.P_lc_nn(input_tensor[[10, 8, 11]])  # FAVD, sza, max_Ac
        P_lt = self.P_lt_nn(input_tensor[[10, 8, 12, 13]])  # FAVD, sza, max_At, mean_At
        # print("P_lc")
        # print(P_lc[test_row, test_col])
        # print("P_lt")
        # print(P_lt[test_row, test_col])
        # print(input_tensor[[4,5,10,11,12],test_row,test_col])
        # print("small kc kt kz kg",input_tensor[0:4, test_row, test_col])
        # print("kc kt kz kg",input_tensor[4:8, test_row, test_col])

        # return P_lt.unsqueeze(0)
        # 计算BRF分量
        cos_sza = torch.cos(input_tensor[8])
        # P_lc=1
        # SF = 3.14*SF
        # P_lt=1
        # SF=input_tensor[14:14 + self.bandnum * 1]
        BRF_lc = (input_tensor[0] * input_tensor[10] * SF * P_lc) / (cos_sza)
        BRF_lt = (input_tensor[1] * input_tensor[10] * SF * P_lt) / (cos_sza)

        dim_len = len(BRF_lc.shape)
        if dim_len == 2:
            BRF_l = (BRF_lc+BRF_lt).unsqueeze(0)
        else:
            BRF_l = BRF_lc + BRF_lt
        BRF_single = BRF_l + BRF_s

        # return BRF_l

        # 收集第一部分结果
        components.extend([BRF_s, BRF_l, BRF_single])

        # ------ 第二部分计算结果（使用big参数）------
        BRF_s_big = input_tensor[7] * input_tensor[14 + 2 * self.bandnum:14 + 3 * self.bandnum]

        BRF_lc_big = (input_tensor[4] * input_tensor[10] * SF * P_lc) / (cos_sza)
        BRF_lt_big = (input_tensor[5] * input_tensor[10] * SF * P_lt) / (cos_sza )


        dim_len = len(BRF_lc_big.shape)
        if dim_len == 2:
            BRF_l_big = (BRF_lc_big + BRF_lt_big).unsqueeze(0)
        else:
            BRF_l_big = BRF_lc_big + BRF_lt_big
        BRF_single_big = BRF_l_big + BRF_s_big

        # 收集第二部分结果
        components.extend([BRF_s_big, BRF_l_big, BRF_single_big])
        # print("BRF_single_big",BRF_single_big.shape)
        # print(BRF_single_big[:,test_row,test_col])

        # for _ in components:
        #     print(_.shape)

        # 合并所有结果（避免任何切片赋值）
        return BRF_lc_big,input_tensor[4:8]
        # return torch.cat(components, dim=0)  # 最终形状 [6*bandnum, H, W]


    def forward_inc_sgl(self, input_tensor):
        # 生成所有结果组件（避免任何原地操作）
        components = []
        test_row=0
        test_col=0

        # ------ 第一部分计算结果 ------
        BRF_s = input_tensor[3] * input_tensor[14 + 2 * self.bandnum:14 + 3 * self.bandnum]

        # 计算中间变量
        part1 = input_tensor[[8, 9]]  # sza, saa
        part2 = input_tensor[14:14 + self.bandnum * 2]  # rl, tl
        SF = self.SF_nn(torch.cat([part1, part2], dim=0))
        # print("SF")
        # print(SF[:,test_row,test_col])


        P_lc = self.P_lc_nn(input_tensor[[10, 8, 11]])  # FAVD, sza, max_Ac
        P_lt = self.P_lt_nn(input_tensor[[10, 8, 12, 13]])  # FAVD, sza, max_At, mean_At
        # print("P_lc")
        # print(P_lc[test_row, test_col])
        # print("P_lt")
        # print(P_lt[test_row, test_col])
        # print(input_tensor[[4,5,10,11,12],test_row,test_col])
        # print("small kc kt kz kg",input_tensor[0:4, test_row, test_col])
        # print("kc kt kz kg",input_tensor[4:8, test_row, test_col])

        # return P_lt.unsqueeze(0)
        # 计算BRF分量
        cos_sza = torch.cos(input_tensor[8])
        # P_lc=1
        # SF = 3.14*SF
        # P_lt=1
        # SF=input_tensor[14:14 + self.bandnum * 1]
        # BRF_lc = (input_tensor[0] * input_tensor[10] * SF * P_lc) / (cos_sza  **  2)
        # BRF_lt = (input_tensor[1] * input_tensor[10] * SF * P_lt) / (cos_sza  **  3)
        BRF_lc = (input_tensor[0] * input_tensor[10] * SF * P_lc) / (cos_sza)
        BRF_lt = (input_tensor[1] * input_tensor[10] * SF * P_lt) / (cos_sza)

        dim_len = len(BRF_lc.shape)
        if dim_len == 2:
            BRF_l = (BRF_lc+BRF_lt).unsqueeze(0)
        else:
            BRF_l = BRF_lc + BRF_lt
        BRF_single = BRF_l + BRF_s

        # return BRF_l

        # 收集第一部分结果
        components.extend([BRF_s, BRF_l, BRF_single])

        # ------ 第二部分计算结果（使用big参数）------
        BRF_s_big = input_tensor[7] * input_tensor[14 + 2 * self.bandnum:14 + 3 * self.bandnum]

        # BRF_lc_big = (input_tensor[4] * input_tensor[10] * SF * P_lc) / (cos_sza ** 2)
        # BRF_lt_big = (input_tensor[5] * input_tensor[10] * SF * P_lt) / (cos_sza ** 3)
        BRF_lc_big = (input_tensor[4] * input_tensor[10] * SF * P_lc) / (cos_sza)
        BRF_lt_big = (input_tensor[5] * input_tensor[10] * SF * P_lt) / (cos_sza)

        # BRF_lc_big = (input_tensor[4] * input_tensor[10] * SF * P_lc) / (cos_sza  **  2)* input_tensor[11]
        # BRF_lt_big = (input_tensor[5] * input_tensor[10] * SF * P_lt) / (cos_sza  **  3)* input_tensor[13]
        # print(test)

        # print("BRF_lc_big",BRF_lc_big.shape)
        # print(BRF_lc_big[:,test_row,test_col])
        # print("BRF_lt_big",BRF_lt_big.shape)
        # print(BRF_lt_big[:,test_row,test_col])
        # print("cos_sza",cos_sza[test_row,test_col])

        dim_len = len(BRF_lc_big.shape)
        if dim_len == 2:
            BRF_l_big = (BRF_lc_big + BRF_lt_big).unsqueeze(0)
        else:
            BRF_l_big = BRF_lc_big + BRF_lt_big
        BRF_single_big = BRF_l_big + BRF_s_big

        # 收集第二部分结果
        components.extend([BRF_s_big, BRF_l_big, BRF_single_big])
        # print("BRF_single_big",BRF_single_big.shape)
        # print(BRF_single_big[:,test_row,test_col])

        # for _ in components:
        #     print(_.shape)

        # 合并所有结果（避免任何切片赋值）
        return BRF_single,BRF_single_big
        # return torch.cat(components, dim=0)  # 最终形状 [6*bandnum, H, W]
