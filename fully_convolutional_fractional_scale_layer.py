# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
# import time
import matplotlib.pyplot as plt
import torchvision as torchvision
import torch
import torchvision
from typing import Optional, List, Tuple, Union
import numpy as np
import torch.nn.functional as F
class FullyConvolutionalFractionalScaling2D(torch.nn.Module):
    def compute_padding(self, kernel_size: int)->int:
        compute_padding_from_k = lambda x: (x//2)-1 if (x%2==0) else x//2
        padding = [compute_padding_from_k(k) for k in kernel_size] if isinstance(kernel_size, (list, tuple, np.ndarray)) else compute_padding_from_k(kernel_size)
        return padding

    def fill_weights_NN(self)->torch.Tensor:
        # filter values filling
        kernel_size = self.s
        weight = np.zeros([self.r**2,1,1, kernel_size, kernel_size])
        a = self.s/self.r
        nearest_neighbours = np.round(np.linspace(a/2,(self.s-1)-(a/2),self.r)).astype(int)
        nx, ny = np.meshgrid(nearest_neighbours,nearest_neighbours)
        for r_i, (y, x) in enumerate(zip(nx.flatten(), ny.flatten())):
            weight[r_i,0, 0, x, y] = 1.

        weight = torch.Tensor(weight)
        return weight


    def fill_weights_BiLinear(self)->torch.Tensor:
        kernel_size = self.s
        weight = np.zeros([self.r**2,1,1, kernel_size, kernel_size])
        a = self.s/self.r
        nearest_neighbours = np.linspace(a/2,(self.s-1)-(a/2),self.r)
        nx, ny = np.meshgrid(nearest_neighbours,nearest_neighbours)
        for r_i, (y, x) in enumerate(zip(nx.flatten(), ny.flatten())):
            x_0, x_1 = np.floor(x).astype(int), np.ceil(x).astype(int)
            y_0, y_1 = np.floor(y).astype(int), np.ceil(y).astype(int)
            if x_1 == x_0:
                x_1 += 1
            if y_1 == y_0:
                y_1 += 1

            weight[r_i,0, 0, x_0, y_0] = (x_1-x)*(y_1-y)
            if y_1 < kernel_size:
                weight[r_i,0, 0, x_0, y_1] = (x_1-x)*(y-y_0)
            if x_1 < kernel_size:
                weight[r_i,0, 0, x_1, y_0] = (x-x_0)*(y_1-y)
            if (x_1 < kernel_size) and (y_1 < kernel_size):
                weight[r_i,0, 0, x_1, y_1] = (x-x_0)*(y-y_0)

        weight = torch.Tensor(weight)
        return weight

    def fill_weights_BiQubic(self)->torch.Tensor:
        kernel_size = self.s+1
        weight = np.zeros([self.r**2,1,1, kernel_size, kernel_size])
        a = self.s/self.r
        nearest_neighbours = np.linspace(a/2,(self.s-1)-(a/2),self.r)
        nx, ny = np.meshgrid(nearest_neighbours,nearest_neighbours)
        def get_weight(t,s):
            a = -0.5
            x = np.abs(t-s)
            if np.abs((t-s)==0):
                return 1
            elif 0 < x and x <= 1:
                return (a+2)*(np.abs(t-s)**3)-(a+3)*((t-s)**2)+1
            elif 1 < x and x < 2:
                return (a)*(np.abs(t-s)**3)-(5*a)*((t-s)**2)+(8*a)*np.abs(t-s)-(4*a)
            else:
                return 0
        for r_i, (y, x) in enumerate(zip(nx.flatten(), ny.flatten())):
            x_0, x_1, x_2, x_3 = np.floor(x).astype(int)-1, np.floor(x).astype(int), np.ceil(x).astype(int),np.ceil(x).astype(int)+1
            y_0, y_1, y_2, y_3 = np.floor(y).astype(int)-1, np.floor(y).astype(int), np.ceil(y).astype(int),np.ceil(y).astype(int)+1
            x_0, y_0 = np.max([0,x_0]), np.max([0,y_0])
            x_3, y_3 = np.min([kernel_size-1,x_3]), np.min([kernel_size-1,y_3])

            Ax = np.zeros((1,4))
            Ax[0,2] = get_weight(x_2,x)
            Ax[0,1] = get_weight(x_1,x)
            Ax[0,3] = get_weight(x_3,x)
            Ax[0,0] = get_weight(x_0,x)

            Ay = np.zeros((1,4))
            Ay[0,2] = get_weight(y_2,y)
            Ay[0,1] = get_weight(y_1,y)
            Ay[0,3] = get_weight(y_3,y)
            Ay[0,0] = get_weight(y_0,y)

            W = Ay.T@Ax

            if (x_3 < kernel_size) and (0<=x_0) and (y_3 < kernel_size) and (0 <= y_0):
                weight[r_i,0, 0, x_0:x_3+1, y_0:y_3+1] = W[0:x_3-x_0+1,0:y_3-y_0+1] / ((W[0:x_3-x_0+1,0:y_3-y_0+1]).sum() if (W[0:x_3-x_0+1,0:y_3-y_0+1]).sum() > 0 else 1)
            elif (x_3 < kernel_size) and (0<=x_0) and (y_3 < kernel_size+1) and (0 <= y_0):
                weight[r_i,0, 0, x_0:x_3+1, y_0:y_2+1] = W[:,y_0:y_2+1]/(W[:,y_0:y_2+1]).sum()
            elif (x_3 < kernel_size+1) and (0<=x_0) and (y_3 < kernel_size) and (0 < y_0):
                weight[r_i,0, 0, x_0:x_2+1, y_0:y_3+1] = W[x_0:x_2+1,:]/(W[x_0:x_2+1,:]).sum()
            elif (x_3 < kernel_size+1) and (0<=x_0) and (y_3 < kernel_size+1) and (0 <= y_0):
                weight[r_i,0, 0, x_0:x_2+1, y_0:y_2+1] = W[x_0:x_2+1,y_0:y_2+1]/(W[x_0:x_2+1,y_0:y_2+1]).sum()
            elif (x_3 < kernel_size) and (0<=x_0) and (y_3 < kernel_size+2) and (0 <= y_0):
                weight[r_i,0, 0, x_0:x_3+1, y_0:y_1+1] = W[:,y_0:y_1+1]/(W[:,y_0:y_1+1]).sum()
            elif (x_3 < kernel_size+2) and (0<=x_0) and (y_3 < kernel_size) and (0 <=y_0):
                weight[r_i,0, 0, x_0:x_1+1, y_0:y_3+1] = W[x_0:x_1+1,:]/(W[x_0:x_1+1,:]).sum()
            elif (x_3 < kernel_size+1) and (0<=x_0) and (y_3 < kernel_size+2) and (0 <= y_0):
                weight[r_i,0, 0, x_0:x_2+1, y_0:y_1+1] = W[x_0:x_2+1,y_0:y_1+1]/(W[x_0:x_2+1,y_0:y_1+1]).sum()
            elif (x_3 < kernel_size+2) and (0<=x_0) and (y_3 < kernel_size+1) and (0 <=y_0):
                weight[r_i,0, 0, x_0:x_1+1, y_0:y_2+1] = W[x_0:x_1+1,y_0:y_2+1]/(W[x_0:x_1+1,y_0:y_2+1]).sum()
            elif (x_3 < kernel_size+2) and (0<=x_0) and (y_3 < kernel_size+2) and (0 <= y_0):
                weight[r_i,0, 0, x_0:x_1+1, y_0:y_1+1] = W[x_0:x_1+1,y_0:y_1+1]/(W[x_0:x_1+1,y_0:y_1+1]).sum()
            else:
                weight[r_i,0, 0, x_1, y_1]=1
        weight = torch.Tensor(weight, device=self.device)
        return weight


    def __init__(self,
                 r:             int,
                 s:             int,
                 scaling_mode: str='nearest',
                 is_inner_layer: bool=False) -> None:
        super(FullyConvolutionalFractionalScaling2D, self).__init__()
        self.r = r
        self.s = s
        self.K = 8
        self.is_inner_layer = is_inner_layer
        self.scaling_modes = {
            'bicubic':  self.fill_weights_BiQubic,
            'nearest':  self.fill_weights_NN,
            'bilinear': self.fill_weights_BiLinear
        }
        self.pooling_kernel = self.scaling_modes[scaling_mode]()
        self.gaussian_blur = torchvision.transforms.functional.gaussian_blur
        self.pixelshuffle = torch.nn.PixelShuffle(self.r)#self.fill_weights_BiLInear(r, s)
    # def pixelshuffle(self, input: torch.Tensor) -> torch.Tensor:
    #     H,W = input.shape[-2:]
    #     H_n = (H//self.K)+1 if H%self.K != 0 else (H//self.K)
    #     W_n = (W//self.K)+1 if W%self.K != 0 else (W//self.K)
    #     res = []#[[None for j in range(0,W_n)] for i in range(0,H_n)]
    #     for i in range(0, H_n):
    #         res.append([])
    #         for j in range(0, W_n):
    #             res[-1].append(F.pixel_shuffle(input[..., i * self.K:(i + 1) * self.K, j * self.K:(j + 1) * self.K], self.r))
    #             torch.cuda.empty_cache()
    #         res[-1] = torch.concat(res[-1],dim=-1)
    #         torch.cuda.empty_cache()
    #     res = torch.concat(res,dim=-2)
    #     return res
    def conv3d(self,input):
        kernel_size = self.s+1
        padding_shape = self.compute_padding(kernel_size)
        padding = F.pad(input,pad=[padding_shape,padding_shape,padding_shape,padding_shape,0,0],mode='replicate')
        filtered_input = F.conv3d(padding, \
                                      self.pooling_kernel.to(input.device), \
                                      stride=[1,self.s,self.s], \
                                      padding='valid', \
                                      bias=None)
        return filtered_input
    def encode_image(self, input: torch.Tensor):
        if not self.is_inner_layer:
            if len(input.shape) == 3:
                input = input[None, ...]
            x = torch.permute(input, (0,3,1,2))
        else:
            x = input
        return x
    def decode_image(self, x, input: torch.Tensor):
        if not self.is_inner_layer:
            res = torch.permute(x, (0,2,3,1))
            if len(input.shape) == 3:
                res = res[0]
        else:
            res = x
        return res

    def forward(self,input: torch.Tensor) -> torch.Tensor:
        x = self.encode_image(input)
        # torch.cuda.empty_cache()
        # kernel_size = max(3,self.s-2 if self.s%2!=0 else self.s-1)
        # x = self.gaussian_blur(x,kernel_size)
        x = x[:, None, :, :, :]
        # torch.cuda.empty_cache()
        x = self.conv3d(x)
        # torch.cuda.empty_cache()
        x = torch.permute(x, (0,2,1,3,4))
        x = self.pixelshuffle(x)
        # torch.cuda.empty_cache()
        x = torch.squeeze(x, 2)
        res = self.decode_image(x, input)
        return res


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     import numpy as np
#     A = cv2.imread(r"C:\Users\micha\Pictures\Scrrenshots\FN\Screenshot 2021-12-06 232355 - Copy.png")
#     times = []
#
# FCFS0 = FullyConvolutionalFractionalScaling2D(r=3,s=2,scaling_mode='bicubic') # downsampling by factor 2/3
#     FCFS1 = FullyConvolutionalFractionalScaling2D(r=23,s=5,scaling_mode='bilinear') # downsampling by factor 2/3
#     FCFS2 = FullyConvolutionalFractionalScaling2D(r=23,s=3,scaling_mode='bicubic') # downsampling by factor 2/3

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
