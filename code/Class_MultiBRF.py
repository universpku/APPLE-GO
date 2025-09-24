import torch
import torch.nn as nn
import numpy as np
import math

class MultiBRF_nn_mulband(nn.Module):
    def __init__(self,size_x,g,bandnum):
        super(MultiBRF_nn_mulband, self).__init__()
        self.G=g
        self.bandnum=bandnum
        self.size_x=size_x
        self.x_list=(torch.arange(0,1,1.0/size_x)*np.pi/2).reshape(-1,1,1)
        self.cos_x=torch.cos(self.x_list)
        self.sin_x=torch.sin(self.x_list)
        self.multiBRF_adj_nn=nn.Conv2d(1, 1, kernel_size=(3, 3), bias=False, stride=1,padding=1)
        w1=(4.0*math.sqrt(2.0)+4)
        w2=(4.0+2*math.sqrt(2.0))
        self.multiBRF_adj_nn.weight.data=torch.tensor([[[[1./w1,1/w2,1/w1],
                                                        [1/w2,0,1/w2],
                                                        [1/w1,1/w2,1/w1]]]])


    def forward(self,input_tensor):
        '''
        :param input_tensor:
            0:FAVD,1:sza,2:LAI,
            3:3+self.bandnum:rl
            3+self.bandnum:3+2*self.bandnum:tl
            3+2*self.bandnum:3+3*self.bandnum:rs
            3+3*self.bandnum:3+4*self.bandnum:belta
        :return:
        '''


        cos_sza=torch.cos(input_tensor[1,:,:])
        input_tensor[2, :, :]=(input_tensor[2,:,:]+1e-8)*0.3
        Is = 1-torch.exp(-self.G*input_tensor[2,:,:]/torch.cos(input_tensor[1,:,:]))
        Id = 2 * (self.sin_x*self.cos_x*(1-torch.exp(-self.G*input_tensor[2,:,:]/self.cos_x))).sum(dim=0)/self.size_x*np.pi/2

        # k1 = 0.0045 * torch.exp(1.2555 * torch.cos(input_tensor[1,:,:]))
        # k2 = 0.1982 * torch.log(torch.cos(input_tensor[1,:,:])) - 0.7146
        # p = 0.7 * torch.exp(k1 * input_tensor[2,:,:]) - 0.66 * torch.exp(k2 * input_tensor[2,:,:])

        cos_sza = torch.cos(input_tensor[1,:,:])
        part1 = 0.059 * cos_sza + 0.741
        part2 = (0.274 * cos_sza - 0.961) *  input_tensor[2,:,:]
        p = part1 * (1 - torch.exp(part2))

        I0 = input_tensor[3+3*self.bandnum:3+4*self.bandnum,:,:] * Id + (1 - input_tensor[3+3*self.bandnum:3+4*self.bandnum,:,:]) * Is  # 冠层拦截之和（对直射+散射）
        w = input_tensor[3:3+self.bandnum,:,:] + input_tensor[3+self.bandnum:3+2*self.bandnum,:,:]  # 单次散射反照率
        BRF_ml = I0 * w * p * w * Is / 2 / input_tensor[2,:,:] / (1 - p * w) / np.pi  # 除以Π，来自统一模型公式

        # 再计算BRF_ms 来自土壤的多次散射贡献量-----Li,RSE,2024 & Zeng,2018
        Rdn = Id * w * Id / input_tensor[2,:,:] / 2 / (1 - p * w)  # 土壤反照率(各向同性)
        Tdn = torch.exp(-1*self.G * input_tensor[2,:,:] / torch.cos(input_tensor[1,:,:])) + I0 * w * Id / input_tensor[2,:,:] / 2 / (1 - p * w)
        Tup = torch.exp(-1*self.G * input_tensor[2,:,:]) + Id * w * Id / input_tensor[2,:,:] / 2 / (1 - p * w)
        BRF_ms = (input_tensor[3+2*self.bandnum:3+3*self.bandnum, :, :] * Tdn * Tup / (1 - input_tensor[3+2*self.bandnum:3+3*self.bandnum, :, :] * Rdn) - torch.exp(
            -1 * self.G * input_tensor[2, :, :] / torch.cos(input_tensor[1, :, :])) * input_tensor[3+2*self.bandnum:3+3*self.bandnum, :, :] * torch.exp(
            -1* self.G * input_tensor[2, :, :])) / 2 / np.pi  # 除以2Π，来自统一模型公式
        BRF_multi = BRF_ml + BRF_ms

        scale=self.multiBRF_adj_nn(input_tensor[2,:,:].reshape(1,1,input_tensor.shape[1],input_tensor.shape[2]))
        # print(input_tensor[0,:,:])
        # print(scale)
        scale=scale[0,0,:,:]/input_tensor[2,:,:]+1

        # result_tensor 包含BRF_ml, BRF_ms, BRF_multi
        result_tensor=torch.zeros([4*self.bandnum,input_tensor.shape[1],input_tensor.shape[2]])
        result_tensor[0:self.bandnum,:,:]=BRF_ml
        result_tensor[1*self.bandnum:2*self.bandnum,:,:]=BRF_ms
        result_tensor[2*self.bandnum:3*self.bandnum,:,:]=BRF_multi
        result_tensor[3*self.bandnum:4*self.bandnum,:,:]=scale*BRF_multi

        # print("Begin in MultiBRF_nn_mulband")
        # print("input_tensor[:, 1, 1]")
        # print(input_tensor[:, 1, 1])
        # print("result_tensor[:, 1, 1]")
        # print(result_tensor[:, 1, 1])
        # print("scale")
        # print(scale[1, 1])
        # print("End in MultiBRF_nn_mulband")

        return scale*BRF_multi
        # return result_tensor

    def forward_just(self,input_tensor):
        '''
        :param input_tensor:
            0:FAVD,1:sza,2:LAI,
            3:3+self.bandnum:rl
            3+self.bandnum:3+2*self.bandnum:tl
            3+2*self.bandnum:3+3*self.bandnum:rs
            3+3*self.bandnum:3+4*self.bandnum:belta
        :return:
        '''
        cos_sza=torch.cos(input_tensor[1,:,:])
        input_tensor[2, :, :]=input_tensor[2,:,:]+1e-8
        Is = 1-torch.exp(-self.G*input_tensor[2,:,:]/torch.cos(input_tensor[1,:,:]))
        Id = 2 * (self.sin_x*self.cos_x*(1-torch.exp(-self.G*input_tensor[2,:,:]/self.cos_x))).sum(dim=0)/self.size_x*np.pi/2

        k1 = 0.0045 * torch.exp(1.2555 * torch.cos(input_tensor[1,:,:]))
        k2 = 0.1982 * torch.log(torch.cos(input_tensor[1,:,:])) - 0.7146
        p = 0.7 * torch.exp(k1 * input_tensor[2,:,:]) - 0.66 * torch.exp(k2 * input_tensor[2,:,:])
        I0 = input_tensor[3+3*self.bandnum:3+4*self.bandnum,:,:] * Id + (1 - input_tensor[3+3*self.bandnum:3+4*self.bandnum,:,:]) * Is  # 冠层拦截之和（对直射+散射）
        w = input_tensor[3:3+self.bandnum,:,:] + input_tensor[3+self.bandnum:3+2*self.bandnum,:,:]  # 单次散射反照率
        BRF_ml = I0 * w * p * w * Is / 2 / input_tensor[2,:,:] / (1 - p * w) / np.pi  # 除以Π，来自统一模型公式

        # 再计算BRF_ms 来自土壤的多次散射贡献量-----Li,RSE,2024 & Zeng,2018
        Rdn = Id * w * Id / input_tensor[2,:,:] / 2 / (1 - p * w)  # 土壤反照率(各向同性)
        Tdn = torch.exp(-1*self.G * input_tensor[2,:,:] / torch.cos(input_tensor[1,:,:])) + I0 * w * Id / input_tensor[2,:,:] / 2 / (1 - p * w)
        Tup = torch.exp(-1*self.G * input_tensor[2,:,:]) + Id * w * Id / input_tensor[2,:,:] / 2 / (1 - p * w)
        BRF_ms = (input_tensor[3+2*self.bandnum:3+3*self.bandnum, :, :] * Tdn * Tup / (1 - input_tensor[3+2*self.bandnum:3+3*self.bandnum, :, :] * Rdn) - torch.exp(
            -1 * self.G * input_tensor[2, :, :] / torch.cos(input_tensor[1, :, :])) * input_tensor[3+2*self.bandnum:3+3*self.bandnum, :, :] * torch.exp(
            -1* self.G * input_tensor[2, :, :])) / 2 / np.pi  # 除以2Π，来自统一模型公式
        BRF_multi = BRF_ml + BRF_ms

        scale=self.multiBRF_adj_nn(input_tensor[2,:,:].reshape(1,1,input_tensor.shape[1],input_tensor.shape[2]))
        scale=scale[0,0,:,:]/input_tensor[2,:,:]+1

        # result_tensor 包含BRF_ml, BRF_ms, BRF_multi
        result_tensor=torch.zeros([4*self.bandnum,input_tensor.shape[1],input_tensor.shape[2]])
        result_tensor[0:self.bandnum,:,:]=BRF_ml
        result_tensor[1*self.bandnum:2*self.bandnum,:,:]=BRF_ms
        result_tensor[2*self.bandnum:3*self.bandnum,:,:]=BRF_multi
        result_tensor[3*self.bandnum:4*self.bandnum,:,:]=scale*BRF_multi
        return BRF_multi

    def forward_inc_sgl(self,input_tensor):
        '''
        :param input_tensor:
            0:FAVD,1:sza,2:LAI,
            3:3+self.bandnum:rl
            3+self.bandnum:3+2*self.bandnum:tl
            3+2*self.bandnum:3+3*self.bandnum:rs
            3+3*self.bandnum:3+4*self.bandnum:belta
        :return:
        '''


        cos_sza=torch.cos(input_tensor[1,:,:])
        input_tensor[2, :, :]=input_tensor[2,:,:]+1e-8
        Is = 1-torch.exp(-self.G*input_tensor[2,:,:]/torch.cos(input_tensor[1,:,:]))
        Id = 2 * (self.sin_x*self.cos_x*(1-torch.exp(-self.G*input_tensor[2,:,:]/self.cos_x))).sum(dim=0)/self.size_x*np.pi/2

        # k1 = 0.0045 * torch.exp(1.2555 * torch.cos(input_tensor[1,:,:]))
        # k2 = 0.1982 * torch.log(torch.cos(input_tensor[1,:,:])) - 0.7146
        # p = 0.7 * torch.exp(k1 * input_tensor[2,:,:]) - 0.66 * torch.exp(k2 * input_tensor[2,:,:])
        cos_sza = torch.cos(input_tensor[1, :, :])
        part1 = 0.059 * cos_sza + 0.741
        part2 = (0.274 * cos_sza - 0.961) * input_tensor[2, :, :]
        p = part1 * (1 - torch.exp(part2))
        # print("forward_inc_sgl in MultiBRF_nn_mulband")

        I0 = input_tensor[3+3*self.bandnum:3+4*self.bandnum,:,:] * Id + (1 - input_tensor[3+3*self.bandnum:3+4*self.bandnum,:,:]) * Is  # 冠层拦截之和（对直射+散射）
        w = input_tensor[3:3+self.bandnum,:,:] + input_tensor[3+self.bandnum:3+2*self.bandnum,:,:]  # 单次散射反照率
        BRF_ml = I0 * w * p * w * Is / 2 / input_tensor[2,:,:] / (1 - p * w) / np.pi  # 除以Π，来自统一模型公式

        # 再计算BRF_ms 来自土壤的多次散射贡献量-----Li,RSE,2024 & Zeng,2018
        Rdn = Id * w * Id / input_tensor[2,:,:] / 2 / (1 - p * w)  # 土壤反照率(各向同性)
        Tdn = torch.exp(-1*self.G * input_tensor[2,:,:] / torch.cos(input_tensor[1,:,:])) + I0 * w * Id / input_tensor[2,:,:] / 2 / (1 - p * w)
        Tup = torch.exp(-1*self.G * input_tensor[2,:,:]) + Id * w * Id / input_tensor[2,:,:] / 2 / (1 - p * w)


        BRF_ms = (input_tensor[3+2*self.bandnum:3+3*self.bandnum, :, :] * Tdn * Tup / (1 - input_tensor[3+2*self.bandnum:3+3*self.bandnum, :, :] * Rdn) - torch.exp(
            -1 * self.G * input_tensor[2, :, :] / torch.cos(input_tensor[1, :, :])) * input_tensor[3+2*self.bandnum:3+3*self.bandnum, :, :] * torch.exp(
            -1* self.G * input_tensor[2, :, :])) / 2 / np.pi  # 除以2Π，来自统一模型公式
        BRF_multi = BRF_ml + BRF_ms

        scale=self.multiBRF_adj_nn(input_tensor[2,:,:].reshape(1,1,input_tensor.shape[1],input_tensor.shape[2]))
        # print(input_tensor[0,:,:])
        # print(scale)
        scale=scale[0,0,:,:]/input_tensor[2,:,:]+1

        # result_tensor 包含BRF_ml, BRF_ms, BRF_multi
        result_tensor=torch.zeros([4*self.bandnum,input_tensor.shape[1],input_tensor.shape[2]])
        result_tensor[0:self.bandnum,:,:]=BRF_ml
        result_tensor[1*self.bandnum:2*self.bandnum,:,:]=BRF_ms
        result_tensor[2*self.bandnum:3*self.bandnum,:,:]=BRF_multi
        result_tensor[3*self.bandnum:4*self.bandnum,:,:]=scale*BRF_multi

        # print("Begin in MultiBRF_nn_mulband")
        # print("input_tensor[:, 1, 1]")
        # print(input_tensor[:, 1, 1])
        # print("result_tensor[:, 1, 1]")
        # print(result_tensor[:, 1, 1])
        # print("scale")
        # print(scale[1, 1])
        # print("End in MultiBRF_nn_mulband")

        return BRF_multi,scale*BRF_multi
        # return result_tensor[2*self.bandnum:3*self.bandnum,:,:]*0,scale*BRF_multi
        # return result_tensor
