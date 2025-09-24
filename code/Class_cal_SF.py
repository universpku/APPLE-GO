import torch
import torch.nn as nn
import numpy as np

class cal_SF_mulband(nn.Module):
    '''
    计算SF的神经网络
    '''
    def __init__(self,size_x,size_y,bandnum):
        super(cal_SF_mulband, self).__init__()
        self.size_x=size_x
        self.size_y=size_y
        self.bandnum=bandnum
        x_list=torch.arange(np.pi/2/size_x/2,np.pi/2,np.pi/2/size_x)
        y_list=torch.arange(np.pi/size_y,2*np.pi,2*np.pi/size_y)
        # x_list = torch.arange(0, np.pi / 2+1e-8, np.pi / 2 / size_x)
        # y_list = torch.arange(0, 2 * np.pi+1e-8, 2 * np.pi / size_y)
        x_list = x_list.reshape(-1, 1)
        y_list = y_list.reshape(1, -1)
        sin_x=torch.sin(x_list)
        cos_x=torch.cos(x_list)
        # print("sin_x",sin_x)
        # print("cos_x",cos_x)
        sin_y=torch.sin(y_list)
        cos_y=torch.cos(y_list)
        self.cal_tensor_1=sin_x*cos_x*sin_x*cos_y
        self.cal_tensor_2 = sin_x * cos_x *  sin_x * sin_y
        self.cal_tensor_3 = sin_x * cos_x *  cos_x
        self.cal_tensor_1=self.cal_tensor_1.reshape(self.cal_tensor_1.shape[0],self.cal_tensor_1.shape[1],1,1)
        self.cal_tensor_2 = self.cal_tensor_2.reshape(self.cal_tensor_2.shape[0], self.cal_tensor_2.shape[1], 1, 1)
        self.cal_tensor_3 = self.cal_tensor_3.reshape(self.cal_tensor_3.shape[0], self.cal_tensor_3.shape[1], 1, 1)
        # print(self.cal_tensor_1.shape)


    def forward(self, input_tensor):
        '''
        计算SF的函数
        :param input_tensor: 一个[4,n,n]的矩阵，
            0：sza.
            1:saa,
            2:2+self.bandnum:rl,
            2+self.bandnum:2+self.bandnum*2:tl
        :return:
        '''
        sin_sza=torch.sin(input_tensor[0,:,:])
        cos_sza=torch.cos(input_tensor[0,:,:])
        sin_saa=torch.sin(input_tensor[1,:,:])
        cos_saa=torch.cos(input_tensor[1,:,:])
        result_tensor=self.cal_tensor_1*sin_sza*cos_saa+self.cal_tensor_2*sin_sza*sin_saa+self.cal_tensor_3*cos_sza
        result_tensor_1=torch.max(result_tensor,torch.zeros_like(result_tensor)).unsqueeze(2)
        result_tensor_2=torch.max(-result_tensor,torch.zeros_like(result_tensor)).unsqueeze(2)
        # print(result_tensor_1.shape)
        # print(result_tensor_2.shape)
        # print("result_tensor_1",result_tensor_1.sum(dim=(0,1)))
        # print("result_tensor_2",result_tensor_2.sum(dim=(0,1)))

        # result_tensor=result_tensor_1*input_tensor[2+self.bandnum:2+self.bandnum*2,:,:]+result_tensor_2*input_tensor[2:2+self.bandnum,:,:]
        result_tensor = result_tensor_2 * input_tensor[2 + self.bandnum:2 + self.bandnum * 2, :,
                                          :] + result_tensor_1 * input_tensor[2:2 + self.bandnum, :, :]

        # print(result_tensor.shape)
        result_tensor=result_tensor.sum(dim=(0,1))/ 2 * np.pi/result_tensor.shape[0]/result_tensor.shape[1]
        #
        return result_tensor