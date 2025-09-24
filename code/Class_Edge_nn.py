import torch
import torch.nn as nn

def DecEdge_pnj(matrix):
    '''
    边缘检测，pnj修改版本
    使用torch.nn.Conv2d实现
    :param matrix:
    :return:
    '''
    matrix_new=torch.Tensor(matrix.copy()).unsqueeze(0).unsqueeze(0)
    matrix_new[matrix_new>0]=1
    # matrix = torch.Tensor(matrix)
    # print(matrix.shape)
    # corr2d=nn.Conv2d(1,1,kernel_size=(3,3),bias=False,stride=1,padding=1,padding_mode='circular')
    corr2d = nn.Conv2d(1, 1, kernel_size=(3, 3), bias=False, stride=1)
    corr2d.weight.data=torch.tensor([[[[1.,1,1],
                                      [1,0,1],
                                      [1,1,1]]]])
    # corr2d.weight.data = torch.tensor([[[[1., 0, 0],
    #                                      [0, 0, 0],
    #                                      [0, 0, 0]]]])
    # print(corr2d.weight.data.shape)
    matrix_torch=8-corr2d(matrix_new)
    matrix_torch[matrix_torch <= 1] = 0
    matrix_torch[matrix_torch>1]=1
    matrix_torch[matrix_new[:,:,1:-1,1:-1] == 0] = 0
    # print(matrix_torch.shape)
    return matrix_torch.sum(),matrix_torch.detach().numpy()[0,0]

class Edge_nn_tensor(nn.Module):
    '''
    边缘检测的神经网络
    '''
    def __init__(self,block_size_x,block_size_y):
        super(Edge_nn_tensor, self).__init__()
        self.block_size_x=block_size_x
        self.block_size_y=block_size_y
        self.corrd2d=nn.Conv2d(1,1,kernel_size=(3,3),stride=1)
        self.corrd2d.weight=torch.nn.Parameter(torch.tensor([[[[-1.,-1,-1],
                                        [-1,0,-1],
                                        [-1,-1,-1]]]]), requires_grad=False)
        self.corrd2d.bias= torch.nn.Parameter(torch.tensor([8.0]), requires_grad=False)

    def forward(self, input_tensor):
        '''
        计算边缘检测的函数
        :param input_tensor: 需要是一个[1,n,n]的张量，CHM
        :return: [2,n,n]的张量，0:Count_edge,1:Gap_count
        '''
        input_tensor[input_tensor < 0] = 0
        matrix_torch=DecEdge_pnj(input_tensor)
        return matrix_torch