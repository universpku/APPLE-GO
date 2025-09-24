import torch
import torch.nn as nn

from Class_Upsample import DynamicBlockUpsample2D
from read_and_write_tiff import read_tiff
from Class_Single_k import Single_k_nn_0303
from Class_BRF_nn import BRF_nn
from Class_cal_k import cal_k


class BRFModel(nn.Module):
    def __init__(self, bandNumber, **kwargs):
        super(BRFModel, self).__init__()
        self.bandNumber = bandNumber
        self.size_x = kwargs.get("size_x", 30)  # 一个像元x方向（）包含多少个chm像元点
        self.size_y = kwargs.get("size_y", 30)  # 一个像元y方向（）包含多少个chm像元点

        self.chm_path = kwargs.get("chm_path", "input/CHM.tif")
        self.path1_path = kwargs.get("path1_path", "input/PATH1.tif")
        self.path2_path = kwargs.get("path2_path", "input/PATH2.tif")
        self.CHM = self.load_raster_to_nograd_parameter(self.chm_path)
        # self.PATH1 = self.load_raster_to_parameter(self.path1_path)
        # self.PATH2 = self.load_raster_to_parameter(self.path2_path)
        self.PATH1 = self.load_raster_to_nograd_parameter(self.path1_path)
        self.PATH2 = self.load_raster_to_nograd_parameter(self.path2_path)

        init_TH = kwargs.get("init_TH", 10.0)
        Boolean_TH = kwargs.get("Boolean_TH", False)
        if (isinstance(init_TH, (int, float))):
            self.TH = nn.Parameter(
                torch.full((self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y), init_TH,
                           dtype=torch.float32), requires_grad=Boolean_TH)
        elif (isinstance(init_TH, str)):
            self.TH = self.load_th_to_parameter(init_TH, Boolean_TH)
        self.FAVD = nn.Parameter(torch.full((self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), kwargs.get("init_FAVD", 0.5), dtype=torch.float32))
        self.dL = nn.Parameter(torch.full((self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), kwargs.get("init_dL", 0.1), dtype=torch.float32),requires_grad=False)

        # self.TH = ConstrainedParameter(torch.full((self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y),
        #                                   kwargs.get("init_TH", 10.0), dtype=torch.float32),constraint_func=lambda x: x.clamp_(min=0))
        # self.FAVD = ConstrainedParameter(torch.full((self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y),
        #                                     kwargs.get("init_FAVD", 0.5), dtype=torch.float32),constraint_func=lambda x: x.clamp_(min=0))
        # self.dL = ConstrainedParameter(
        #     torch.full((self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y), kwargs.get("init_dL", 0.1),
        #                dtype=torch.float32),constraint_func=lambda x: x.clamp_(min=0))

        init_rs = kwargs.get("init_rs", 0.2)
        if isinstance(init_rs, (int, float)):
            self.rs = nn.Parameter(torch.full((1, self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), init_rs, dtype=torch.float32),requires_grad=False)
            # self.rs = nn.Parameter(torch.full((1, self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y), init_rs,
            #                dtype=torch.float32))

            # self.rs = ConstrainedParameter(torch.full((1, self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y), init_rs,
            #                dtype=torch.float32),constraint_func=lambda x: x.clamp_(min=0))
        elif isinstance(init_rs, list):
            self.rs = nn.Parameter(torch.stack([torch.full((self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), val , dtype=torch.float32) for val in init_rs]),requires_grad=False)
            # self.rs = nn.Parameter(torch.stack([torch.full((self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y), val, dtype=torch.float32) for val
            #                                     in init_rs]))

            # self.rs = ConstrainedParameter(torch.stack([torch.full(
            #     (self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y), val, dtype=torch.float32) for val
            #                                     in init_rs]),constraint_func=lambda x: x.clamp_(min=0))

        init_rl = kwargs.get("init_rl", 0.5)
        if isinstance(init_rl, (int, float)):
            self.rl = nn.Parameter(torch.full((1, self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), init_rl, dtype=torch.float32),requires_grad=False)
            # self.rl = nn.Parameter(torch.full((1, self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y), init_rl,
            #                dtype=torch.float32))
            # self.rl = ConstrainedParameter(
            #     torch.full((1, self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y), init_rl,
            #                dtype=torch.float32),constraint_func=lambda x: x.clamp_(min=0))
        elif isinstance(init_rl, list):
            self.rl = nn.Parameter(torch.stack([torch.full((self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), val , dtype=torch.float32) for val in init_rl]),requires_grad=False)
            # self.rl = nn.Parameter(torch.stack([torch.full((self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y), val, dtype=torch.float32) for val
            #                                     in init_rl]))

            # self.rl = ConstrainedParameter(torch.stack([torch.full(
            #     (self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y), val, dtype=torch.float32) for val
            #                                     in init_rl]),constraint_func=lambda x: x.clamp_(min=0))
        init_tl = kwargs.get("init_tl", 0.4)
        if isinstance(init_tl, (int, float)):
            self.tl = nn.Parameter(torch.full((1, self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), init_tl, dtype=torch.float32),requires_grad=False)
            # self.tl = nn.Parameter(torch.full((1, self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y), init_tl,
            #                dtype=torch.float32))

            # self.tl = ConstrainedParameter(
            #     torch.full((1, self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y), init_tl,
            #                dtype=torch.float32),constraint_func=lambda x: x.clamp_(min=0))
        elif isinstance(init_tl, list):
            self.tl = nn.Parameter(torch.stack([torch.full((self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), val , dtype=torch.float32) for val in init_tl]),requires_grad=False)
            # self.tl = nn.Parameter(torch.stack([torch.full((self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y), val, dtype=torch.float32) for val
            #                                     in init_tl]))

            # self.tl = ConstrainedParameter(torch.stack([torch.full(
            #     (self.CHM.shape[0] // self.size_x, self.CHM.shape[1] // self.size_y), val, dtype=torch.float32) for val
            #     in init_tl]),constraint_func=lambda x: x.clamp_(min=0))
        init_beta = kwargs.get("init_beta", 0.4)
        if isinstance(init_beta, (int, float)):
            self.belta = nn.Parameter(torch.full((1, self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), init_beta, dtype=torch.float32), requires_grad=False)
        elif isinstance(init_beta, list):
            self.belta = nn.Parameter(torch.stack([torch.full((self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), val , dtype=torch.float32) for val in init_beta]), requires_grad=False)


        # self.rs = nn.Parameter(torch.full((1,self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), kwargs.get("init_rs", 0.2), dtype=torch.float32))
        # self.rl = nn.Parameter(torch.full((1,self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), kwargs.get("init_rl", 0.5), dtype=torch.float32))
        # self.tl = nn.Parameter(torch.full((1,self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), kwargs.get("init_tl", 0.4), dtype=torch.float32))
        # self.belta = nn.Parameter(torch.full((1,self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), kwargs.get("init_belta", 0.4), dtype=torch.float32))



        self.sza = nn.Parameter(torch.full((self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), kwargs.get("init_sza", 0.0), dtype=torch.float32), requires_grad=False)
        self.saa = nn.Parameter(torch.full((self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), kwargs.get("init_saa", 0.0), dtype=torch.float32), requires_grad=False)
        self.LAI = nn.Parameter(torch.full((self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y), kwargs.get("init_LAI", 0.0), dtype=torch.float32), requires_grad=False)
        # self.TH = nn.Parameter(torch.tensor(kwargs.get("init_TH", 10.0), dtype=torch.float32))
        # self.FAVD = nn.Parameter(torch.tensor(kwargs.get("init_FAVD", 0.5), dtype=torch.float32))
        # self.dL = nn.Parameter(torch.tensor(kwargs.get("init_dL", 0.1), dtype=torch.float32))
        # self.rs = nn.Parameter(torch.tensor(kwargs.get("init_rs", 0.2), dtype=torch.float32))

        self.up2D=DynamicBlockUpsample2D(self.size_x)
        self.single_k_nn = Single_k_nn_0303(self.size_x, self.size_y)
        self.BRF_nn = BRF_nn(self.size_x, self.size_y,0.5,self.dL,self.bandNumber)

        self.Edge_nn = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1)
        self.Edge_nn.weight = torch.nn.Parameter(torch.tensor([[[[-1., -1, -1],
                                                                 [-1, 0, -1],
                                                                 [-1, -1, -1]]]]), requires_grad=False)
        self.Edge_nn.bias = torch.nn.Parameter(torch.tensor([8.0]), requires_grad=False)
        self.tensor_CHM = torch.Tensor(self.CHM.clone()).unsqueeze(0).unsqueeze(0)
        self.tensor_CHM[self.tensor_CHM > 0] = 1
        self.tensor_edge = torch.zeros([self.CHM.shape[0], self.CHM.shape[1]])
        self.tensor_edge[1:-1, 1:-1] = self.Edge_nn(self.tensor_CHM)
        # print("self.tensor_edge.shape", self.tensor_edge.shape)
        self.tensor_edge[self.tensor_edge <= 1] = 0
        self.tensor_edge[self.tensor_edge > 1] = 1
        self.tensor_edge[self.tensor_CHM[0,0] == 0] = 0

        self.tensor_CHM[0, 0, :, :][self.CHM > 0] = 2
        self.tensor_CHM[0,0, :, :][self.tensor_edge == 1] = 1

        num_block_x = int(self.CHM.shape[0] / self.size_x)
        num_block_y = int(self.CHM.shape[1] / self.size_y)
        self.Count_edge = self.tensor_edge.reshape(1, num_block_x, self.size_x, num_block_y, self.size_y).sum(
            dim=(2, 4))
        self.Gap_count = (2 - self.tensor_CHM).reshape(1, num_block_x, self.size_x, num_block_y, self.size_y).sum(
            dim=(2, 4)) / 2
        # print("self.Count_edge.shape", self.Count_edge.shape)
        # print("self.Gap_count.shape", self.Gap_count.shape)

        self.cal_k_nn = cal_k(self.size_x, self.size_y)



    def load_raster_to_parameter(self, filepath):
        """加载栅格数据并转换为可训练参数"""
        data = read_tiff(filepath)
        param = nn.Parameter(torch.tensor(data, dtype=torch.float32), requires_grad=True)
        return param

    def load_raster_to_nograd_parameter(self, filepath):
        """加载栅格数据"""
        data = read_tiff(filepath)
        param = nn.Parameter(torch.tensor(data, dtype=torch.float32), requires_grad=False)
        return param

    def forward(self):
        # 第一步将FAVD，TH,sza上采用至CHM的大小
        FAVD_upsampled=self.up2D(self.FAVD.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        TH_upsampled=self.up2D(self.TH.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        sza_upsampled=self.up2D(self.sza.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        # 第二步将CHM，PATH1，PATH2，FAVD，TH，CHM.max,sza拼接，并输入single_k_nn
        CHM_max = torch.full_like(self.CHM, self.CHM.max(), dtype=torch.float32)
        input_array = torch.stack([self.CHM,
                                   self.PATH1,
                                   self.PATH2,
                                   FAVD_upsampled,
                                   TH_upsampled,
                                   CHM_max,
                                   sza_upsampled,
                                   self.tensor_edge], dim=0)
        # input_array = torch.cat([self.CHM.unsqueeze(0),self.PATH1.unsqueeze(0), self.PATH2.unsqueeze(0), FAVD_upsampled.unsqueeze(0), TH_upsampled.unsqueeze(0), CHM_max.unsqueeze(0), sza_upsampled.unsqueeze(0),self.tensor_edge.unsqueeze(0)], dim=0)
        # print("input_array.shape",input_array.shape)

        tensor_edge = self.single_k_nn(input_array)
        # print("tensor_edge.shape", tensor_edge.shape)
        # print("self.Count_edge.shape", self.Count_edge.shape)
        # print("self.Gap_count.shape", self.Gap_count.shape)
        # print("self.sza.shape", self.sza.shape)
        tensor_input = torch.cat([
            tensor_edge[0:7],  # 前7个通道 [7, H, W]
            self.Count_edge,  # 添加通道维度 [1, H, W]
            self.Gap_count,  # [1, H, W]
            self.sza.unsqueeze(0)  # [1, H, W]
        ], dim=0)  # 最终形状 [7+1+1+1=10, H, W]
        # print("tensor_input",tensor_input[:,1,2])

        tensor_result = self.cal_k_nn(tensor_input)
        # print("tensor_result.shape",tensor_result.shape)
        # print("tensor_result", tensor_result[:,1,2])
        # tensor_result_old = self.cal_k_nn.old_forward(tensor_input)
        # print("tensor_result_old.shape",tensor_result_old.shape)
        # compare=tensor_result[1]-tensor_result_old[1]
        # print("tensor_result[0]",tensor_result[1])
        # print("compare",compare)
        # print("compare.max",compare.max())

        # return tensor_result

        # tensor_part2=torch.zeros([15+self.bandNumber*4,self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y])
        # tensor_part2[0:8,:,:]=tensor_result[0:8,:,:]
        # tensor_part2[8,:,:]=self.sza * np.pi / 180
        # tensor_part2[9,:,:]=self.saa * np.pi / 180
        # tensor_part2[10,:,:]=self.FAVD
        # # tensor_part2[11:15,:,:]=tensor_edge[8:12,:,:]
        # tensor_part2[11:14, :, :] = tensor_edge[10:13, :, :]
        # tensor_part2[14, :, :] = tensor_edge[7, :, :]
        # tensor_part2[15:15+self.bandNumber,:,:]=self.rl
        # tensor_part2[15+self.bandNumber:15+self.bandNumber*2,:,:]=self.tl
        # tensor_part2[15+self.bandNumber*2:15+self.bandNumber*3,:,:]=self.rs
        # tensor_part2[15+self.bandNumber*3:15+self.bandNumber*4,:,:]=self.belta
        # print(tensor_part2[:,1,2])

        # 计算基础维度
        chm_rows = self.CHM.shape[0] // self.size_x
        chm_cols = self.CHM.shape[1] // self.size_y

        # 分块处理各部分数据
        components = [
            # 第一部分: 0-7通道
            tensor_result[0:8],  # [8, H, W]

            # 第二部分: 角度转换 (8-10通道)
            torch.stack([
                self.sza * torch.pi / 180,
                self.saa * torch.pi / 180,
                self.FAVD
            ], dim=0),  # [3, H, W]

            # 第三部分: Edge相关 (11-14通道)
            torch.cat([
                tensor_edge[10:13],  # [3, H, W]
                tensor_edge[7:8]  # [1, H, W]
            ], dim=0),  # 总[4, H, W]

            # 第四部分: 波段相关数据 (15+band*4通道)
            torch.cat([
                self.rl,
                self.tl,
                self.rs,
                self.belta
            ], dim=0)  # [band*4, H, W]
        ]

        # for _ in components:
        #     print(_.shape)
        # 拼接所有组件
        tensor_part2 = torch.cat(components, dim=0)
        # print("in BRFmodel forward")
        # print(tensor_part2[:,1,1])

        tensor_result=self.BRF_nn(tensor_part2)
        # print(tensor_result[:,1,2])



        # 第三步将

        return tensor_result

    def constrained_parameter(self):
        self.TH.data = self.TH.data.clamp(min=0)
        self.FAVD.data = self.FAVD.data.clamp(min=0)
        self.dL.data = self.dL.data.clamp(min=0)

        self.rs.data = self.rs.data.clamp(min=0)
        self.rl.data = self.rl.data.clamp(min=0)
        self.tl.data = self.tl.data.clamp(min=0)

    def load_th_to_parameter(self,filepath,boolean_th):
        data =  read_tiff(filepath)
        num_block_x = int(data.shape[0] / self.size_x)
        num_block_y = int(data.shape[1] / self.size_y)
        data2=data.reshape(num_block_x, self.size_x, num_block_y, self.size_y).mean(axis=(1, 3))
        param = nn.Parameter(torch.tensor(data2, dtype=torch.float32), requires_grad=boolean_th)
        return param

    def show_FAVD(self):
        print(self.FAVD)
        return self.FAVD

    def show_TH(self):
        print(self.TH)
        return self.TH

    def show_rs(self):
        print(self.rs)
        return self.rs

    def show_rl(self):
        print(self.rl)
        return self.rl

    def show_tl(self):
        print(self.tl)
        return self.tl

    def get_FAVD(self):
        return self.FAVD

    def get_TH(self):
        return self.TH

    def get_rs(self):
        return self.rs

    def get_rl(self):
        return self.rl

    def get_tl(self):
        return self.tl

    def set_rl(self,rl):
        self.rl=rl

    def set_tl(self,tl):
        self.tl=tl

    def set_FAVD(self, FAVD):
        requires_grad= self.FAVD.requires_grad
        self.FAVD = nn.Parameter(torch.zeros_like(self.FAVD, dtype=torch.float32)+ FAVD, requires_grad=requires_grad)
        # print(self.FAVD.dtype)

    def apart_forward(self):
        # 第一步将FAVD，TH,sza上采用至CHM的大小
        FAVD_upsampled = self.up2D(self.FAVD.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        TH_upsampled = self.up2D(self.TH.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        sza_upsampled = self.up2D(self.sza.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        # 第二步将CHM，PATH1，PATH2，FAVD，TH，CHM.max,sza拼接，并输入single_k_nn
        CHM_max = torch.full_like(self.CHM, self.CHM.max(), dtype=torch.float32)
        input_array = torch.stack([self.CHM,
                                   self.PATH1,
                                   self.PATH2,
                                   FAVD_upsampled,
                                   TH_upsampled,
                                   CHM_max,
                                   sza_upsampled,
                                   self.tensor_edge], dim=0)
        # input_array = torch.cat([self.CHM.unsqueeze(0),self.PATH1.unsqueeze(0), self.PATH2.unsqueeze(0), FAVD_upsampled.unsqueeze(0), TH_upsampled.unsqueeze(0), CHM_max.unsqueeze(0), sza_upsampled.unsqueeze(0),self.tensor_edge.unsqueeze(0)], dim=0)
        # print("input_array.shape",input_array.shape)

        tensor_edge = self.single_k_nn(input_array)
        # print("tensor_edge.shape", tensor_edge.shape)
        # print("self.Count_edge.shape", self.Count_edge.shape)
        # print("self.Gap_count.shape", self.Gap_count.shape)
        # print("self.sza.shape", self.sza.shape)
        tensor_input = torch.cat([
            tensor_edge[0:7],  # 前7个通道 [7, H, W]
            self.Count_edge,  # 添加通道维度 [1, H, W]
            self.Gap_count,  # [1, H, W]
            self.sza.unsqueeze(0)  # [1, H, W]
        ], dim=0)  # 最终形状 [7+1+1+1=10, H, W]
        # print("tensor_input",tensor_input[:,1,2])

        tensor_result = self.cal_k_nn(tensor_input)
        # print("tensor_result.shape",tensor_result.shape)
        # print("tensor_result", tensor_result[:,1,2])
        # tensor_result_old = self.cal_k_nn.old_forward(tensor_input)
        # print("tensor_result_old.shape",tensor_result_old.shape)
        # compare=tensor_result[1]-tensor_result_old[1]
        # print("tensor_result[0]",tensor_result[1])
        # print("compare",compare)
        # print("compare.max",compare.max())

        # return tensor_result

        # tensor_part2=torch.zeros([15+self.bandNumber*4,self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y])
        # tensor_part2[0:8,:,:]=tensor_result[0:8,:,:]
        # tensor_part2[8,:,:]=self.sza * np.pi / 180
        # tensor_part2[9,:,:]=self.saa * np.pi / 180
        # tensor_part2[10,:,:]=self.FAVD
        # # tensor_part2[11:15,:,:]=tensor_edge[8:12,:,:]
        # tensor_part2[11:14, :, :] = tensor_edge[10:13, :, :]
        # tensor_part2[14, :, :] = tensor_edge[7, :, :]
        # tensor_part2[15:15+self.bandNumber,:,:]=self.rl
        # tensor_part2[15+self.bandNumber:15+self.bandNumber*2,:,:]=self.tl
        # tensor_part2[15+self.bandNumber*2:15+self.bandNumber*3,:,:]=self.rs
        # tensor_part2[15+self.bandNumber*3:15+self.bandNumber*4,:,:]=self.belta
        # print(tensor_part2[:,1,2])

        # 计算基础维度
        chm_rows = self.CHM.shape[0] // self.size_x
        chm_cols = self.CHM.shape[1] // self.size_y

        # 分块处理各部分数据
        components = [
            # 第一部分: 0-7通道
            tensor_result[0:8],  # [8, H, W]

            # 第二部分: 角度转换 (8-10通道)
            torch.stack([
                self.sza * torch.pi / 180,
                self.saa * torch.pi / 180,
                self.FAVD
            ], dim=0),  # [3, H, W]

            # 第三部分: Edge相关 (11-14通道)
            torch.cat([
                tensor_edge[10:13],  # [3, H, W]
                tensor_edge[7:8]  # [1, H, W]
            ], dim=0),  # 总[4, H, W]

            # 第四部分: 波段相关数据 (15+band*4通道)
            torch.cat([
                self.rl,
                self.tl,
                self.rs,
                self.belta
            ], dim=0)  # [band*4, H, W]
        ]

        # for _ in components:
        #     print(_.shape)
        # 拼接所有组件
        tensor_part2 = torch.cat(components, dim=0)
        # print(tensor_part2[:,1,2])


        tensor_first=self.BRF_nn.just_first(tensor_part2)

        tensor_multi=self.BRF_nn.just_multi(tensor_part2)



        # 第三步将

        return tensor_first,tensor_multi

    def apart_forward_more(self):
        # 第一步将FAVD，TH,sza上采用至CHM的大小
        FAVD_upsampled = self.up2D(self.FAVD.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        TH_upsampled = self.up2D(self.TH.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        sza_upsampled = self.up2D(self.sza.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        # 第二步将CHM，PATH1，PATH2，FAVD，TH，CHM.max,sza拼接，并输入single_k_nn
        CHM_max = torch.full_like(self.CHM, self.CHM.max(), dtype=torch.float32)
        input_array = torch.stack([self.CHM,
                                   self.PATH1,
                                   self.PATH2,
                                   FAVD_upsampled,
                                   TH_upsampled,
                                   CHM_max,
                                   sza_upsampled,
                                   self.tensor_edge], dim=0)
        # input_array = torch.cat([self.CHM.unsqueeze(0),self.PATH1.unsqueeze(0), self.PATH2.unsqueeze(0), FAVD_upsampled.unsqueeze(0), TH_upsampled.unsqueeze(0), CHM_max.unsqueeze(0), sza_upsampled.unsqueeze(0),self.tensor_edge.unsqueeze(0)], dim=0)
        # print("input_array.shape",input_array.shape)
        # print("here")
        tensor_edge = self.single_k_nn(input_array)
        # print("here?")
        # print("tensor_edge.shape", tensor_edge.shape)
        # print("self.Count_edge.shape", self.Count_edge.shape)
        # print("self.Gap_count.shape", self.Gap_count.shape)
        # print("self.sza.shape", self.sza.shape)
        tensor_input = torch.cat([
            tensor_edge[0:7],  # 前7个通道 [7, H, W]
            self.Count_edge,  # 添加通道维度 [1, H, W]
            self.Gap_count,  # [1, H, W]
            self.sza.unsqueeze(0)  # [1, H, W]
        ], dim=0)  # 最终形状 [7+1+1+1=10, H, W]
        # print("tensor_input",tensor_input[:,1,2])
        # print("here???")
        tensor_result = self.cal_k_nn(tensor_input)
        # print("tensor_result.shape",tensor_result.shape)
        # print("tensor_result", tensor_result[:,1,2])
        # tensor_result_old = self.cal_k_nn.old_forward(tensor_input)
        # print("tensor_result_old.shape",tensor_result_old.shape)
        # compare=tensor_result[1]-tensor_result_old[1]
        # print("tensor_result[0]",tensor_result[1])
        # print("compare",compare)
        # print("compare.max",compare.max())

        # return tensor_result

        # tensor_part2=torch.zeros([15+self.bandNumber*4,self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y])
        # tensor_part2[0:8,:,:]=tensor_result[0:8,:,:]
        # tensor_part2[8,:,:]=self.sza * np.pi / 180
        # tensor_part2[9,:,:]=self.saa * np.pi / 180
        # tensor_part2[10,:,:]=self.FAVD
        # # tensor_part2[11:15,:,:]=tensor_edge[8:12,:,:]
        # tensor_part2[11:14, :, :] = tensor_edge[10:13, :, :]
        # tensor_part2[14, :, :] = tensor_edge[7, :, :]
        # tensor_part2[15:15+self.bandNumber,:,:]=self.rl
        # tensor_part2[15+self.bandNumber:15+self.bandNumber*2,:,:]=self.tl
        # tensor_part2[15+self.bandNumber*2:15+self.bandNumber*3,:,:]=self.rs
        # tensor_part2[15+self.bandNumber*3:15+self.bandNumber*4,:,:]=self.belta
        # print(tensor_part2[:,1,2])

        # 计算基础维度
        chm_rows = self.CHM.shape[0] // self.size_x
        chm_cols = self.CHM.shape[1] // self.size_y

        # 分块处理各部分数据
        components = [
            # 第一部分: 0-7通道
            tensor_result[0:8],  # [8, H, W]

            # 第二部分: 角度转换 (8-10通道)
            torch.stack([
                self.sza * torch.pi / 180,
                self.saa * torch.pi / 180,
                self.FAVD
            ], dim=0),  # [3, H, W]

            # 第三部分: Edge相关 (11-14通道)
            torch.cat([
                tensor_edge[10:13],  # [3, H, W]
                tensor_edge[7:8]  # [1, H, W]
            ], dim=0),  # 总[4, H, W]

            # 第四部分: 波段相关数据 (15+band*4通道)
            torch.cat([
                self.rl,
                self.tl,
                self.rs,
                self.belta
            ], dim=0)  # [band*4, H, W]
        ]
        # print("here?????")
        # for _ in components:
        #     print(_.shape)
        # 拼接所有组件
        tensor_part2 = torch.cat(components, dim=0)
        # print(tensor_part2[:,1,2])
        # print(tensor_edge[10])


        tensor_BRF_lc,tensor_BRF_lt=self.BRF_nn.just_first_more(tensor_part2)

        tensor_multi=self.BRF_nn.just_multi(tensor_part2)



        # 第三步将

        return tensor_BRF_lc,tensor_BRF_lt,tensor_multi

    def apart_forward_save(self):
        # 第一步将FAVD，TH,sza上采用至CHM的大小
        FAVD_upsampled = self.up2D(self.FAVD.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        TH_upsampled = self.up2D(self.TH.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        sza_upsampled = self.up2D(self.sza.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        # 第二步将CHM，PATH1，PATH2，FAVD，TH，CHM.max,sza拼接，并输入single_k_nn
        CHM_max = torch.full_like(self.CHM, self.CHM.max(), dtype=torch.float32)
        input_array = torch.stack([self.CHM,
                                   self.PATH1,
                                   self.PATH2,
                                   FAVD_upsampled,
                                   TH_upsampled,
                                   CHM_max,
                                   sza_upsampled,
                                   self.tensor_edge], dim=0)
        # input_array = torch.cat([self.CHM.unsqueeze(0),self.PATH1.unsqueeze(0), self.PATH2.unsqueeze(0), FAVD_upsampled.unsqueeze(0), TH_upsampled.unsqueeze(0), CHM_max.unsqueeze(0), sza_upsampled.unsqueeze(0),self.tensor_edge.unsqueeze(0)], dim=0)
        # print("input_array.shape",input_array.shape)

        tensor_edge = self.single_k_nn(input_array)
        # print("tensor_edge.shape", tensor_edge.shape)
        # print("self.Count_edge.shape", self.Count_edge.shape)
        # print("self.Gap_count.shape", self.Gap_count.shape)
        # print("self.sza.shape", self.sza.shape)
        tensor_input = torch.cat([
            tensor_edge[0:7],  # 前7个通道 [7, H, W]
            self.Count_edge,  # 添加通道维度 [1, H, W]
            self.Gap_count,  # [1, H, W]
            self.sza.unsqueeze(0)  # [1, H, W]
        ], dim=0)  # 最终形状 [7+1+1+1=10, H, W]
        # print("tensor_input",tensor_input[:,1,2])

        tensor_result = self.cal_k_nn(tensor_input)
        # print("tensor_result.shape",tensor_result.shape)
        # print("tensor_result", tensor_result[:,1,2])
        # tensor_result_old = self.cal_k_nn.old_forward(tensor_input)
        # print("tensor_result_old.shape",tensor_result_old.shape)
        # compare=tensor_result[1]-tensor_result_old[1]
        # print("tensor_result[0]",tensor_result[1])
        # print("compare",compare)
        # print("compare.max",compare.max())

        # return tensor_result

        # tensor_part2=torch.zeros([15+self.bandNumber*4,self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y])
        # tensor_part2[0:8,:,:]=tensor_result[0:8,:,:]
        # tensor_part2[8,:,:]=self.sza * np.pi / 180
        # tensor_part2[9,:,:]=self.saa * np.pi / 180
        # tensor_part2[10,:,:]=self.FAVD
        # # tensor_part2[11:15,:,:]=tensor_edge[8:12,:,:]
        # tensor_part2[11:14, :, :] = tensor_edge[10:13, :, :]
        # tensor_part2[14, :, :] = tensor_edge[7, :, :]
        # tensor_part2[15:15+self.bandNumber,:,:]=self.rl
        # tensor_part2[15+self.bandNumber:15+self.bandNumber*2,:,:]=self.tl
        # tensor_part2[15+self.bandNumber*2:15+self.bandNumber*3,:,:]=self.rs
        # tensor_part2[15+self.bandNumber*3:15+self.bandNumber*4,:,:]=self.belta
        # print(tensor_part2[:,1,2])

        # 计算基础维度
        chm_rows = self.CHM.shape[0] // self.size_x
        chm_cols = self.CHM.shape[1] // self.size_y

        # 分块处理各部分数据
        components = [
            # 第一部分: 0-7通道
            tensor_result[0:8],  # [8, H, W]

            # 第二部分: 角度转换 (8-10通道)
            torch.stack([
                self.sza * torch.pi / 180,
                self.saa * torch.pi / 180,
                self.FAVD
            ], dim=0),  # [3, H, W]

            # 第三部分: Edge相关 (11-14通道)
            torch.cat([
                tensor_edge[10:13],  # [3, H, W]
                tensor_edge[7:8]  # [1, H, W]
            ], dim=0),  # 总[4, H, W]

            # 第四部分: 波段相关数据 (15+band*4通道)
            torch.cat([
                self.rl,
                self.tl,
                self.rs,
                self.belta
            ], dim=0)  # [band*4, H, W]
        ]

        # for _ in components:
        #     print(_.shape)
        # 拼接所有组件
        tensor_part2 = torch.cat(components, dim=0)
        # print(tensor_part2[:,1,2])
        print(tensor_edge[10])


        tensor_BRF_lc,tensor_BRF_lt=self.BRF_nn.just_first_more(tensor_part2)

        # tensor_multi=self.BRF_nn.just_multi(tensor_part2)



        # 第三步将

        return tensor_BRF_lc,tensor_BRF_lt,tensor_edge[10:13]

    def apart_forward_small(self):
        # 第一步将FAVD，TH,sza上采用至CHM的大小
        FAVD_upsampled = self.up2D(self.FAVD.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        TH_upsampled = self.up2D(self.TH.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        sza_upsampled = self.up2D(self.sza.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        # 第二步将CHM，PATH1，PATH2，FAVD，TH，CHM.max,sza拼接，并输入single_k_nn
        CHM_max = torch.full_like(self.CHM, self.CHM.max(), dtype=torch.float32)
        input_array = torch.stack([self.CHM,
                                   self.PATH1,
                                   self.PATH2,
                                   FAVD_upsampled,
                                   TH_upsampled,
                                   CHM_max,
                                   sza_upsampled,
                                   self.tensor_edge], dim=0)
        # input_array = torch.cat([self.CHM.unsqueeze(0),self.PATH1.unsqueeze(0), self.PATH2.unsqueeze(0), FAVD_upsampled.unsqueeze(0), TH_upsampled.unsqueeze(0), CHM_max.unsqueeze(0), sza_upsampled.unsqueeze(0),self.tensor_edge.unsqueeze(0)], dim=0)
        # print("input_array.shape",input_array.shape)

        tensor_edge = self.single_k_nn(input_array)
        # print("tensor_edge.shape", tensor_edge.shape)
        # print("self.Count_edge.shape", self.Count_edge.shape)
        # print("self.Gap_count.shape", self.Gap_count.shape)
        # print("self.sza.shape", self.sza.shape)
        tensor_input = torch.cat([
            tensor_edge[0:7],  # 前7个通道 [7, H, W]
            self.Count_edge,  # 添加通道维度 [1, H, W]
            self.Gap_count,  # [1, H, W]
            self.sza.unsqueeze(0)  # [1, H, W]
        ], dim=0)  # 最终形状 [7+1+1+1=10, H, W]
        # print("tensor_input",tensor_input[:,1,2])

        tensor_result = self.cal_k_nn(tensor_input)
        # print("tensor_result.shape",tensor_result.shape)
        # print("tensor_result", tensor_result[:,1,2])
        # tensor_result_old = self.cal_k_nn.old_forward(tensor_input)
        # print("tensor_result_old.shape",tensor_result_old.shape)
        # compare=tensor_result[1]-tensor_result_old[1]
        # print("tensor_result[0]",tensor_result[1])
        # print("compare",compare)
        # print("compare.max",compare.max())

        # return tensor_result

        # tensor_part2=torch.zeros([15+self.bandNumber*4,self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y])
        # tensor_part2[0:8,:,:]=tensor_result[0:8,:,:]
        # tensor_part2[8,:,:]=self.sza * np.pi / 180
        # tensor_part2[9,:,:]=self.saa * np.pi / 180
        # tensor_part2[10,:,:]=self.FAVD
        # # tensor_part2[11:15,:,:]=tensor_edge[8:12,:,:]
        # tensor_part2[11:14, :, :] = tensor_edge[10:13, :, :]
        # tensor_part2[14, :, :] = tensor_edge[7, :, :]
        # tensor_part2[15:15+self.bandNumber,:,:]=self.rl
        # tensor_part2[15+self.bandNumber:15+self.bandNumber*2,:,:]=self.tl
        # tensor_part2[15+self.bandNumber*2:15+self.bandNumber*3,:,:]=self.rs
        # tensor_part2[15+self.bandNumber*3:15+self.bandNumber*4,:,:]=self.belta
        # print(tensor_part2[:,1,2])

        # 计算基础维度
        chm_rows = self.CHM.shape[0] // self.size_x
        chm_cols = self.CHM.shape[1] // self.size_y

        # 分块处理各部分数据
        components = [
            # 第一部分: 0-7通道
            tensor_result[0:8],  # [8, H, W]

            # 第二部分: 角度转换 (8-10通道)
            torch.stack([
                self.sza * torch.pi / 180,
                self.saa * torch.pi / 180,
                self.FAVD
            ], dim=0),  # [3, H, W]

            # 第三部分: Edge相关 (11-14通道)
            torch.cat([
                tensor_edge[10:13],  # [3, H, W]
                tensor_edge[7:8]  # [1, H, W]
            ], dim=0),  # 总[4, H, W]

            # 第四部分: 波段相关数据 (15+band*4通道)
            torch.cat([
                self.rl,
                self.tl,
                self.rs,
                self.belta
            ], dim=0)  # [band*4, H, W]
        ]

        # for _ in components:
        #     print(_.shape)
        # 拼接所有组件
        tensor_part2 = torch.cat(components, dim=0)
        # print(tensor_part2[:,1,2])


        tensor_first=self.BRF_nn.just_first(tensor_part2)

        tensor_multi=self.BRF_nn.just_multi_small(tensor_part2)



        # 第三步将

        return tensor_first,tensor_multi

    def cal_path_path(self,**kwargs):
        '''
        该函数的功能在于计算path/0°时的path，与cos(sita)的差异
        需要注意，由于是区分Ac与At的，因此该计算要到BRF_nn内部写
        :return:
        '''
        path1_bottom_path = kwargs.get("path1_bottom_path", "input/PATH1.tif")
        path2_bottom_path = kwargs.get("path2_bottom_path", "input/PATH2.tif")

        PATH1_bottom = self.load_raster_to_nograd_parameter(path1_bottom_path)
        PATH2_bottom = self.load_raster_to_nograd_parameter(path2_bottom_path)

        # 第一步将FAVD，TH,sza上采用至CHM的大小
        FAVD_upsampled = self.up2D(self.FAVD.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        TH_upsampled = self.up2D(self.TH.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        sza_upsampled = self.up2D(self.sza.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        # 第二步将CHM，PATH1，PATH2，FAVD，TH，CHM.max,sza拼接，并输入single_k_nn
        CHM_max = torch.full_like(self.CHM, self.CHM.max(), dtype=torch.float32)
        input_array = torch.stack([self.CHM,
                                   self.PATH1,
                                   self.PATH2,
                                   FAVD_upsampled,
                                   TH_upsampled,
                                   CHM_max,
                                   sza_upsampled,
                                   self.tensor_edge], dim=0)
        # input_array = torch.cat([self.CHM.unsqueeze(0),self.PATH1.unsqueeze(0), self.PATH2.unsqueeze(0), FAVD_upsampled.unsqueeze(0), TH_upsampled.unsqueeze(0), CHM_max.unsqueeze(0), sza_upsampled.unsqueeze(0),self.tensor_edge.unsqueeze(0)], dim=0)
        # print("input_array.shape",input_array.shape)

        tensor_edge = self.single_k_nn.forward_test52(input_array,PATH2_bottom)
        # print("tensor_edge.shape", tensor_edge.shape)
        # print("self.Count_edge.shape", self.Count_edge.shape)
        # print("self.Gap_count.shape", self.Gap_count.shape)
        # print("self.sza.shape", self.sza.shape)
        tensor_input = torch.cat([
            tensor_edge[0:7],  # 前7个通道 [7, H, W]
            self.Count_edge,  # 添加通道维度 [1, H, W]
            self.Gap_count,  # [1, H, W]
            self.sza.unsqueeze(0)  # [1, H, W]
        ], dim=0)  # 最终形状 [7+1+1+1=10, H, W]
        # print("tensor_input",tensor_input[:,1,2])

        tensor_result = self.cal_k_nn(tensor_input)
        # return tensor_edge[10]

        # print("tensor_result.shape",tensor_result.shape)
        # print("tensor_result", tensor_result[:,1,2])
        # tensor_result_old = self.cal_k_nn.old_forward(tensor_input)
        # print("tensor_result_old.shape",tensor_result_old.shape)
        # compare=tensor_result[1]-tensor_result_old[1]
        # print("tensor_result[0]",tensor_result[1])
        # print("compare",compare)
        # print("compare.max",compare.max())

        # return tensor_result

        # tensor_part2=torch.zeros([15+self.bandNumber*4,self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y])
        # tensor_part2[0:8,:,:]=tensor_result[0:8,:,:]
        # tensor_part2[8,:,:]=self.sza * np.pi / 180
        # tensor_part2[9,:,:]=self.saa * np.pi / 180
        # tensor_part2[10,:,:]=self.FAVD
        # # tensor_part2[11:15,:,:]=tensor_edge[8:12,:,:]
        # tensor_part2[11:14, :, :] = tensor_edge[10:13, :, :]
        # tensor_part2[14, :, :] = tensor_edge[7, :, :]
        # tensor_part2[15:15+self.bandNumber,:,:]=self.rl
        # tensor_part2[15+self.bandNumber:15+self.bandNumber*2,:,:]=self.tl
        # tensor_part2[15+self.bandNumber*2:15+self.bandNumber*3,:,:]=self.rs
        # tensor_part2[15+self.bandNumber*3:15+self.bandNumber*4,:,:]=self.belta
        # print(tensor_part2[:,1,2])

        # 计算基础维度
        chm_rows = self.CHM.shape[0] // self.size_x
        chm_cols = self.CHM.shape[1] // self.size_y

        # 分块处理各部分数据
        components = [
            # 第一部分: 0-7通道
            tensor_result[0:8],  # [8, H, W]

            # 第二部分: 角度转换 (8-10通道)
            torch.stack([
                self.sza * torch.pi / 180,
                self.saa * torch.pi / 180,
                self.FAVD
            ], dim=0),  # [3, H, W]

            # 第三部分: Edge相关 (11-14通道)
            torch.cat([
                tensor_edge[10:13],  # [3, H, W]
                tensor_edge[7:8]  # [1, H, W]
            ], dim=0),  # 总[4, H, W]

            # 第四部分: 波段相关数据 (15+band*4通道)
            torch.cat([
                self.rl,
                self.tl,
                self.rs,
                self.belta
            ], dim=0)  # [band*4, H, W]
        ]

        # for _ in components:
        #     print(_.shape)
        # 拼接所有组件
        tensor_part2 = torch.cat(components, dim=0)
        # print(tensor_part2[:,1,2])
        # print(tensor_edge[10])

        tensor_BRF_lc, tensor_BRF_lt = self.BRF_nn.just_first_more_test52(tensor_part2,tensor_edge[9])

        tensor_multi = self.BRF_nn.just_multi(tensor_part2)

        # 第三步将

        return tensor_BRF_lc, tensor_BRF_lt, tensor_multi

    def apart_forward_kc(self):
        # 第一步将FAVD，TH,sza上采用至CHM的大小
        FAVD_upsampled = self.up2D(self.FAVD.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        TH_upsampled = self.up2D(self.TH.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        sza_upsampled = self.up2D(self.sza.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        # 第二步将CHM，PATH1，PATH2，FAVD，TH，CHM.max,sza拼接，并输入single_k_nn
        CHM_max = torch.full_like(self.CHM, self.CHM.max(), dtype=torch.float32)
        input_array = torch.stack([self.CHM,
                                   self.PATH1,
                                   self.PATH2,
                                   FAVD_upsampled,
                                   TH_upsampled,
                                   CHM_max,
                                   sza_upsampled,
                                   self.tensor_edge], dim=0)
        # input_array = torch.cat([self.CHM.unsqueeze(0),self.PATH1.unsqueeze(0), self.PATH2.unsqueeze(0), FAVD_upsampled.unsqueeze(0), TH_upsampled.unsqueeze(0), CHM_max.unsqueeze(0), sza_upsampled.unsqueeze(0),self.tensor_edge.unsqueeze(0)], dim=0)
        # print("input_array.shape",input_array.shape)
        # print("here")
        tensor_edge = self.single_k_nn(input_array)
        # print("here?")
        # print("tensor_edge.shape", tensor_edge.shape)
        # print("self.Count_edge.shape", self.Count_edge.shape)
        # print("self.Gap_count.shape", self.Gap_count.shape)
        # print("self.sza.shape", self.sza.shape)
        tensor_input = torch.cat([
            tensor_edge[0:7],  # 前7个通道 [7, H, W]
            self.Count_edge,  # 添加通道维度 [1, H, W]
            self.Gap_count,  # [1, H, W]
            self.sza.unsqueeze(0)  # [1, H, W]
        ], dim=0)  # 最终形状 [7+1+1+1=10, H, W]
        # print("tensor_input",tensor_input[:,1,2])
        # print("here???")
        tensor_result = self.cal_k_nn(tensor_input)
        # print("tensor_result.shape",tensor_result.shape)
        # print("tensor_result", tensor_result[:,1,2])
        # tensor_result_old = self.cal_k_nn.old_forward(tensor_input)
        # print("tensor_result_old.shape",tensor_result_old.shape)
        # compare=tensor_result[1]-tensor_result_old[1]
        # print("tensor_result[0]",tensor_result[1])
        # print("compare",compare)
        # print("compare.max",compare.max())

        # return tensor_result

        # tensor_part2=torch.zeros([15+self.bandNumber*4,self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y])
        # tensor_part2[0:8,:,:]=tensor_result[0:8,:,:]
        # tensor_part2[8,:,:]=self.sza * np.pi / 180
        # tensor_part2[9,:,:]=self.saa * np.pi / 180
        # tensor_part2[10,:,:]=self.FAVD
        # # tensor_part2[11:15,:,:]=tensor_edge[8:12,:,:]
        # tensor_part2[11:14, :, :] = tensor_edge[10:13, :, :]
        # tensor_part2[14, :, :] = tensor_edge[7, :, :]
        # tensor_part2[15:15+self.bandNumber,:,:]=self.rl
        # tensor_part2[15+self.bandNumber:15+self.bandNumber*2,:,:]=self.tl
        # tensor_part2[15+self.bandNumber*2:15+self.bandNumber*3,:,:]=self.rs
        # tensor_part2[15+self.bandNumber*3:15+self.bandNumber*4,:,:]=self.belta
        # print(tensor_part2[:,1,2])

        # 计算基础维度
        chm_rows = self.CHM.shape[0] // self.size_x
        chm_cols = self.CHM.shape[1] // self.size_y

        # 分块处理各部分数据
        components = [
            # 第一部分: 0-7通道
            tensor_result[0:8],  # [8, H, W]

            # 第二部分: 角度转换 (8-10通道)
            torch.stack([
                self.sza * torch.pi / 180,
                self.saa * torch.pi / 180,
                self.FAVD
            ], dim=0),  # [3, H, W]

            # 第三部分: Edge相关 (11-14通道)
            torch.cat([
                tensor_edge[10:13],  # [3, H, W]
                tensor_edge[7:8]  # [1, H, W]
            ], dim=0),  # 总[4, H, W]

            # 第四部分: 波段相关数据 (15+band*4通道)
            torch.cat([
                self.rl,
                self.tl,
                self.rs,
                self.belta
            ], dim=0)  # [band*4, H, W]
        ]
        # print("here?????")
        # for _ in components:
        #     print(_.shape)
        # 拼接所有组件
        tensor_part2 = torch.cat(components, dim=0)
        # print(tensor_part2[:,1,2])
        # print(tensor_edge[10])


        BRF_lc,four_component=self.BRF_nn.just_first_kc(tensor_part2)

        tensor_multi=self.BRF_nn.just_multi(tensor_part2)



        # 第三步将

        return BRF_lc,four_component,tensor_multi

    def forward_inc_sgl(self):
        # 第一步将FAVD，TH,sza上采用至CHM的大小
        FAVD_upsampled=self.up2D(self.FAVD.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        TH_upsampled=self.up2D(self.TH.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        sza_upsampled=self.up2D(self.sza.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        # 第二步将CHM，PATH1，PATH2，FAVD，TH，CHM.max,sza拼接，并输入single_k_nn
        CHM_max = torch.full_like(self.CHM, self.CHM.max(), dtype=torch.float32)
        input_array = torch.stack([self.CHM,
                                   self.PATH1,
                                   self.PATH2,
                                   FAVD_upsampled,
                                   TH_upsampled,
                                   CHM_max,
                                   sza_upsampled,
                                   self.tensor_edge], dim=0)
        # input_array = torch.cat([self.CHM.unsqueeze(0),self.PATH1.unsqueeze(0), self.PATH2.unsqueeze(0), FAVD_upsampled.unsqueeze(0), TH_upsampled.unsqueeze(0), CHM_max.unsqueeze(0), sza_upsampled.unsqueeze(0),self.tensor_edge.unsqueeze(0)], dim=0)
        # print("input_array.shape",input_array.shape)

        tensor_edge = self.single_k_nn(input_array)
        # print("tensor_edge.shape", tensor_edge.shape)
        # print("self.Count_edge.shape", self.Count_edge.shape)
        # print("self.Gap_count.shape", self.Gap_count.shape)
        # print("self.sza.shape", self.sza.shape)
        tensor_input = torch.cat([
            tensor_edge[0:7],  # 前7个通道 [7, H, W]
            self.Count_edge,  # 添加通道维度 [1, H, W]
            self.Gap_count,  # [1, H, W]
            self.sza.unsqueeze(0)  # [1, H, W]
        ], dim=0)  # 最终形状 [7+1+1+1=10, H, W]
        # print("tensor_input",tensor_input[:,1,2])

        tensor_result = self.cal_k_nn(tensor_input)
        # print("tensor_result.shape",tensor_result.shape)
        # print("tensor_result", tensor_result[:,1,2])
        # tensor_result_old = self.cal_k_nn.old_forward(tensor_input)
        # print("tensor_result_old.shape",tensor_result_old.shape)
        # compare=tensor_result[1]-tensor_result_old[1]
        # print("tensor_result[0]",tensor_result[1])
        # print("compare",compare)
        # print("compare.max",compare.max())

        # return tensor_result

        # tensor_part2=torch.zeros([15+self.bandNumber*4,self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y])
        # tensor_part2[0:8,:,:]=tensor_result[0:8,:,:]
        # tensor_part2[8,:,:]=self.sza * np.pi / 180
        # tensor_part2[9,:,:]=self.saa * np.pi / 180
        # tensor_part2[10,:,:]=self.FAVD
        # # tensor_part2[11:15,:,:]=tensor_edge[8:12,:,:]
        # tensor_part2[11:14, :, :] = tensor_edge[10:13, :, :]
        # tensor_part2[14, :, :] = tensor_edge[7, :, :]
        # tensor_part2[15:15+self.bandNumber,:,:]=self.rl
        # tensor_part2[15+self.bandNumber:15+self.bandNumber*2,:,:]=self.tl
        # tensor_part2[15+self.bandNumber*2:15+self.bandNumber*3,:,:]=self.rs
        # tensor_part2[15+self.bandNumber*3:15+self.bandNumber*4,:,:]=self.belta
        # print(tensor_part2[:,1,2])

        # 计算基础维度
        chm_rows = self.CHM.shape[0] // self.size_x
        chm_cols = self.CHM.shape[1] // self.size_y

        # 分块处理各部分数据
        components = [
            # 第一部分: 0-7通道
            tensor_result[0:8],  # [8, H, W]

            # 第二部分: 角度转换 (8-10通道)
            torch.stack([
                self.sza * torch.pi / 180,
                self.saa * torch.pi / 180,
                self.FAVD
            ], dim=0),  # [3, H, W]

            # 第三部分: Edge相关 (11-14通道)
            torch.cat([
                tensor_edge[10:13],  # [3, H, W]
                tensor_edge[7:8]  # [1, H, W]
            ], dim=0),  # 总[4, H, W]

            # 第四部分: 波段相关数据 (15+band*4通道)
            torch.cat([
                self.rl,
                self.tl,
                self.rs,
                self.belta
            ], dim=0)  # [band*4, H, W]
        ]

        # for _ in components:
        #     print(_.shape)
        # 拼接所有组件
        tensor_part2 = torch.cat(components, dim=0)
        # print(tensor_part2[:,1,1])

        tensor_result=self.BRF_nn.forward_inc_sgl(tensor_part2)
        # print(tensor_result[:,1,2])



        # 第三步将

        return tensor_result

    def forward_0616(self):
        # 第一步将FAVD，TH,sza上采用至CHM的大小
        FAVD_upsampled=self.up2D(self.FAVD.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        TH_upsampled=self.up2D(self.TH.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        sza_upsampled=self.up2D(self.sza.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        # 第二步将CHM，PATH1，PATH2，FAVD，TH，CHM.max,sza拼接，并输入single_k_nn
        CHM_max = torch.full_like(self.CHM, self.CHM.max(), dtype=torch.float32)
        input_array = torch.stack([self.CHM,
                                   self.PATH1,
                                   self.PATH2,
                                   FAVD_upsampled,
                                   TH_upsampled,
                                   CHM_max,
                                   sza_upsampled,
                                   self.tensor_edge], dim=0)
        # input_array = torch.cat([self.CHM.unsqueeze(0),self.PATH1.unsqueeze(0), self.PATH2.unsqueeze(0), FAVD_upsampled.unsqueeze(0), TH_upsampled.unsqueeze(0), CHM_max.unsqueeze(0), sza_upsampled.unsqueeze(0),self.tensor_edge.unsqueeze(0)], dim=0)
        # print("input_array.shape",input_array.shape)

        tensor_edge = self.single_k_nn(input_array)
        # print("tensor_edge.shape", tensor_edge.shape)
        # print("self.Count_edge.shape", self.Count_edge.shape)
        # print("self.Gap_count.shape", self.Gap_count.shape)
        # print("self.sza.shape", self.sza.shape)
        tensor_input = torch.cat([
            tensor_edge[0:7],  # 前7个通道 [7, H, W]
            self.Count_edge,  # 添加通道维度 [1, H, W]
            self.Gap_count,  # [1, H, W]
            self.sza.unsqueeze(0)  # [1, H, W]
        ], dim=0)  # 最终形状 [7+1+1+1=10, H, W]
        # print("tensor_input",tensor_input[:,1,2])

        tensor_result = self.cal_k_nn(tensor_input)
        # print("tensor_result.shape",tensor_result.shape)
        # print("tensor_result", tensor_result[:,1,2])
        # tensor_result_old = self.cal_k_nn.old_forward(tensor_input)
        # print("tensor_result_old.shape",tensor_result_old.shape)
        # compare=tensor_result[1]-tensor_result_old[1]
        # print("tensor_result[0]",tensor_result[1])
        # print("compare",compare)
        # print("compare.max",compare.max())

        # return tensor_result

        # tensor_part2=torch.zeros([15+self.bandNumber*4,self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y])
        # tensor_part2[0:8,:,:]=tensor_result[0:8,:,:]
        # tensor_part2[8,:,:]=self.sza * np.pi / 180
        # tensor_part2[9,:,:]=self.saa * np.pi / 180
        # tensor_part2[10,:,:]=self.FAVD
        # # tensor_part2[11:15,:,:]=tensor_edge[8:12,:,:]
        # tensor_part2[11:14, :, :] = tensor_edge[10:13, :, :]
        # tensor_part2[14, :, :] = tensor_edge[7, :, :]
        # tensor_part2[15:15+self.bandNumber,:,:]=self.rl
        # tensor_part2[15+self.bandNumber:15+self.bandNumber*2,:,:]=self.tl
        # tensor_part2[15+self.bandNumber*2:15+self.bandNumber*3,:,:]=self.rs
        # tensor_part2[15+self.bandNumber*3:15+self.bandNumber*4,:,:]=self.belta
        # print(tensor_part2[:,1,2])

        # 计算基础维度
        chm_rows = self.CHM.shape[0] // self.size_x
        chm_cols = self.CHM.shape[1] // self.size_y

        # 分块处理各部分数据
        components = [
            # 第一部分: 0-7通道
            tensor_result[0:8],  # [8, H, W]

            # 第二部分: 角度转换 (8-10通道)
            torch.stack([
                self.sza * torch.pi / 180,
                self.saa * torch.pi / 180,
                self.FAVD
            ], dim=0),  # [3, H, W]

            # 第三部分: Edge相关 (11-14通道)
            torch.cat([
                tensor_edge[10:13],  # [3, H, W]
                tensor_edge[7:8]  # [1, H, W]
            ], dim=0),  # 总[4, H, W]

            # 第四部分: 波段相关数据 (15+band*4通道)
            torch.cat([
                self.rl,
                self.tl,
                self.rs,
                self.belta
            ], dim=0)  # [band*4, H, W]
        ]

        # for _ in components:
        #     print(_.shape)
        # 拼接所有组件
        tensor_part2 = torch.cat(components, dim=0)
        # print("in BRFmodel forward_0616")
        # print(tensor_part2[:,1,1])

        BRF_all_self,BRF_all,BRF_single_self,BRF_single=self.BRF_nn.forward_inc_sgl(tensor_part2)
        # print(tensor_result[:,1,2])


        four_component_self = tensor_result[0:4, :, :].clone()  # [4, H, W]
        four_component_adj = tensor_result[4:8, :, :].clone()  # [4, H, W]

        # 第三步将

        return four_component_self,four_component_adj, BRF_all_self, BRF_all, BRF_single_self, BRF_single


    def forward_new_test_0613(self):
        '''
        使用0613的想法进行forward测试，
        :return:
        '''
        # 第一步将FAVD，TH,sza上采用至CHM的大小
        FAVD_upsampled=self.up2D(self.FAVD.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        TH_upsampled=self.up2D(self.TH.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        sza_upsampled=self.up2D(self.sza.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        # 第二步将CHM，PATH1，PATH2，FAVD，TH，CHM.max,sza拼接，并输入single_k_nn
        CHM_max = torch.full_like(self.CHM, self.CHM.max(), dtype=torch.float32)
        input_array = torch.stack([self.CHM,
                                   self.PATH2,
                                   self.PATH2,
                                   FAVD_upsampled,
                                   TH_upsampled,
                                   CHM_max,
                                   sza_upsampled,
                                   self.tensor_edge], dim=0)
        # input_array = torch.cat([self.CHM.unsqueeze(0),self.PATH1.unsqueeze(0), self.PATH2.unsqueeze(0), FAVD_upsampled.unsqueeze(0), TH_upsampled.unsqueeze(0), CHM_max.unsqueeze(0), sza_upsampled.unsqueeze(0),self.tensor_edge.unsqueeze(0)], dim=0)
        # print("input_array.shape",input_array.shape)

        tensor_edge = self.single_k_nn(input_array)
        # print("tensor_edge.shape", tensor_edge.shape)
        # print("self.Count_edge.shape", self.Count_edge.shape)
        # print("self.Gap_count.shape", self.Gap_count.shape)
        # print("self.sza.shape", self.sza.shape)
        tensor_input = torch.cat([
            tensor_edge[0:7],  # 前7个通道 [7, H, W]
            self.Count_edge,  # 添加通道维度 [1, H, W]
            self.Gap_count,  # [1, H, W]
            self.sza.unsqueeze(0)  # [1, H, W]
        ], dim=0)  # 最终形状 [7+1+1+1=10, H, W]
        # print("tensor_input",tensor_input[:,1,2])

        tensor_result = self.cal_k_nn(tensor_input)
        # print("tensor_result.shape",tensor_result.shape)
        # print("tensor_result", tensor_result[:,1,2])
        # tensor_result_old = self.cal_k_nn.old_forward(tensor_input)
        # print("tensor_result_old.shape",tensor_result_old.shape)
        # compare=tensor_result[1]-tensor_result_old[1]
        # print("tensor_result[0]",tensor_result[1])
        # print("compare",compare)
        # print("compare.max",compare.max())

        # return tensor_result

        # tensor_part2=torch.zeros([15+self.bandNumber*4,self.CHM.shape[0]//self.size_x, self.CHM.shape[1]//self.size_y])
        # tensor_part2[0:8,:,:]=tensor_result[0:8,:,:]
        # tensor_part2[8,:,:]=self.sza * np.pi / 180
        # tensor_part2[9,:,:]=self.saa * np.pi / 180
        # tensor_part2[10,:,:]=self.FAVD
        # # tensor_part2[11:15,:,:]=tensor_edge[8:12,:,:]
        # tensor_part2[11:14, :, :] = tensor_edge[10:13, :, :]
        # tensor_part2[14, :, :] = tensor_edge[7, :, :]
        # tensor_part2[15:15+self.bandNumber,:,:]=self.rl
        # tensor_part2[15+self.bandNumber:15+self.bandNumber*2,:,:]=self.tl
        # tensor_part2[15+self.bandNumber*2:15+self.bandNumber*3,:,:]=self.rs
        # tensor_part2[15+self.bandNumber*3:15+self.bandNumber*4,:,:]=self.belta
        # print(tensor_part2[:,1,2])

        # 计算基础维度
        chm_rows = self.CHM.shape[0] // self.size_x
        chm_cols = self.CHM.shape[1] // self.size_y

        # 分块处理各部分数据
        components = [
            # 第一部分: 0-7通道
            tensor_result[0:8],  # [8, H, W]

            # 第二部分: 角度转换 (8-10通道)
            torch.stack([
                self.sza * torch.pi / 180,
                self.saa * torch.pi / 180,
                self.FAVD
            ], dim=0),  # [3, H, W]

            # 第三部分: Edge相关 (11-14通道)
            torch.cat([
                tensor_edge[10:13],  # [3, H, W]
                tensor_edge[7:8]  # [1, H, W]
            ], dim=0),  # 总[4, H, W]

            # 第四部分: 波段相关数据 (15+band*4通道)
            torch.cat([
                self.rl,
                self.tl,
                self.rs,
                self.belta
            ], dim=0)  # [band*4, H, W]
        ]

        # for _ in components:
        #     print(_.shape)
        # 拼接所有组件
        tensor_part2 = torch.cat(components, dim=0)
        # print(tensor_part2[:,1,2])

        BRF_all_self,BRF_all,BRF_single_self,BRF_single=self.BRF_nn.forward_inc_sgl(tensor_part2)
        # print(tensor_result[:,1,2])



        # 第三步将

        return BRF_single_self
