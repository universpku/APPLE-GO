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
        MultiBRF_tensor_part1=input_tensor[[10,8,14],:,:]
        MultiBRF_tensor_part2=input_tensor[15:15+4*self.bandnum,:,:]
        BRF_multi = self.MultiBRF_nn(torch.cat([MultiBRF_tensor_part1,MultiBRF_tensor_part2],dim=0))
        BRF_all=BRF_single+BRF_multi
        return BRF_all
