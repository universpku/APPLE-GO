import torch
import torch.nn as nn
from Class_SingleBRF import SingleBRF_nn_mulband
from Class_MultiBRF import MultiBRF_nn_mulband


class BRF_nn(nn.Module):
    def __init__(self,size_x,size_y,g,bL,bandnum):
        super(BRF_nn, self).__init__()
        self.size_x=size_x
        self.size_y=size_y
        self.G=g
        self.bL=bL
        self.bandnum=bandnum
        self.SingleBRF_nn=SingleBRF_nn_mulband(size_x,size_y,g,bL,bandnum)
        self.MultiBRF_nn=MultiBRF_nn_mulband(size_x,g,bandnum)

    def forward(self,input_tensor):
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
            14:LAI
            15:15+self.bandnum:rl
            15+self.bandnum:15+2*self.bandnum:tl
            15+2*self.bandnum:15+3*self.bandnum:rs
            15+3*self.bandnum:15+4*self.bandnum:belta
        :return:
        '''
        SingleBRF_tensor_part1=input_tensor[0:14,:,:]
        SingleBRF_tensor_part2=input_tensor[15:15+3*self.bandnum,:,:]
        BRF_single = self.SingleBRF_nn(torch.cat([SingleBRF_tensor_part1,SingleBRF_tensor_part2],dim=0))
        # print("BRF_nn forward single")
        # print(BRF_single[:,1,1])
        # print("BRF_single.shape",BRF_single.shape)
        # return BRF_single
        MultiBRF_tensor_part1=input_tensor[[10,8,14],:,:]
        MultiBRF_tensor_part2=input_tensor[15:15+4*self.bandnum,:,:]
        BRF_multi = self.MultiBRF_nn(torch.cat([MultiBRF_tensor_part1,MultiBRF_tensor_part2],dim=0))
        # print("BRF_nn forward multi")
        # print(BRF_multi[:, 1, 1])
        # print("BRF_multi.shape",BRF_multi.shape)
        # print("Begin in BRF_nn")
        # print("BRF_single[:, 2, 8]")
        # print(BRF_single[:,2,8])
        # print("BRF_multi[:, 2, 8]")
        # print(BRF_multi[:,2,8])
        # print("End in BRF_nn")
        BRF_all=BRF_single+BRF_multi
        return BRF_all
        # return torch.cat([BRF_single,BRF_multi],dim=0)

    def forward_inc_sgl(self,input_tensor):
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
            14:LAI
            15:15+self.bandnum:rl
            15+self.bandnum:15+2*self.bandnum:tl
            15+2*self.bandnum:15+3*self.bandnum:rs
            15+3*self.bandnum:15+4*self.bandnum:belta
        :return:
        '''
        SingleBRF_tensor_part1=input_tensor[0:14,:,:]
        SingleBRF_tensor_part2=input_tensor[15:15+3*self.bandnum,:,:]
        BRF_single_self,BRF_single = self.SingleBRF_nn.forward_inc_sgl(torch.cat([SingleBRF_tensor_part1,SingleBRF_tensor_part2],dim=0))
        # print("BRF_nn forward_inc_sgl single")
        # print(BRF_single[:,1,1])
        # print("BRF_single.shape",BRF_single.shape)
        # return BRF_single
        MultiBRF_tensor_part1=input_tensor[[10,8,14],:,:]
        MultiBRF_tensor_part2=input_tensor[15:15+4*self.bandnum,:,:]
        BRF_multi_self,BRF_multi = self.MultiBRF_nn.forward_inc_sgl(torch.cat([MultiBRF_tensor_part1,MultiBRF_tensor_part2],dim=0))
        # print("BRF_nn forward_inc_sgl multi")
        # print(BRF_multi[:, 1, 1])
        # print("BRF_multi.shape",BRF_multi.shape)
        # print("Begin in BRF_nn")
        # print("BRF_single[:, 2, 8]")
        # print(BRF_single[:,2,8])
        # print("BRF_multi[:, 2, 8]")
        # print(BRF_multi[:,2,8])
        # print("End in BRF_nn")
        BRF_all_self=BRF_single_self+BRF_multi_self
        BRF_all=BRF_single+BRF_multi
        return BRF_all_self,BRF_all,BRF_single_self,BRF_single
        # return torch.cat([BRF_single,BRF_multi],dim=0)
    def just_first(self,input_tensor):
        SingleBRF_tensor_part1 = input_tensor[0:14, :, :]
        SingleBRF_tensor_part2 = input_tensor[15:15 + 3 * self.bandnum, :, :]
        BRF_single = self.SingleBRF_nn(torch.cat([SingleBRF_tensor_part1, SingleBRF_tensor_part2], dim=0))
        return BRF_single
    def just_multi(self,input_tensor):
        MultiBRF_tensor_part1 = input_tensor[[10, 8, 14], :, :]
        MultiBRF_tensor_part2 = input_tensor[15:15 + 4 * self.bandnum, :, :]
        BRF_multi = self.MultiBRF_nn(torch.cat([MultiBRF_tensor_part1, MultiBRF_tensor_part2], dim=0))
        # print("BRF_multi.shape",BRF_multi)
        return BRF_multi

    def just_first_more(self,input_tensor):
        SingleBRF_tensor_part1 = input_tensor[0:14, :, :]
        SingleBRF_tensor_part2 = input_tensor[15:15 + 3 * self.bandnum, :, :]
        BRF_lc,BRF_lt = self.SingleBRF_nn.forward_more(torch.cat([SingleBRF_tensor_part1, SingleBRF_tensor_part2], dim=0))
        return BRF_lc,BRF_lt

    def just_first_more_test52(self,input_tensor,lrmean):
        print("just_first_more_test52")
        print(lrmean)
        SingleBRF_tensor_part1 = input_tensor[0:14, :, :]
        SingleBRF_tensor_part2 = input_tensor[15:15 + 3 * self.bandnum, :, :]
        BRF_lc,BRF_lt = self.SingleBRF_nn.forward_more_test52(torch.cat([SingleBRF_tensor_part1, SingleBRF_tensor_part2], dim=0),lrmean)
        return BRF_lc,BRF_lt

    def just_multi_small(self,input_tensor):
        MultiBRF_tensor_part1 = input_tensor[[10, 8, 14], :, :]
        MultiBRF_tensor_part2 = input_tensor[15:15 + 4 * self.bandnum, :, :]
        BRF_multi = self.MultiBRF_nn.forward_just(torch.cat([MultiBRF_tensor_part1, MultiBRF_tensor_part2], dim=0))
        return BRF_multi

    def just_first_kc(self,input_tensor):
        SingleBRF_tensor_part1 = input_tensor[0:14, :, :]
        SingleBRF_tensor_part2 = input_tensor[15:15 + 3 * self.bandnum, :, :]
        BRF,four_comp = self.SingleBRF_nn.forward_kc(torch.cat([SingleBRF_tensor_part1, SingleBRF_tensor_part2], dim=0))
        return BRF,four_comp