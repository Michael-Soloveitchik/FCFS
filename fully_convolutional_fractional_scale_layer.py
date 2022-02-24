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

class FullyConvolutionalFractionalScaling2D(torch.nn.Module):
    def compute_padding(self, kernel_size: int)->int:
        compute_padding_from_k = lambda x: x//2-1 if (x%2==0) else x//2
        padding = [compute_padding_from_k(k) for k in kernel_size] if isinstance(kernel_size, (list, tuple, np.ndarray)) else compute_padding_from_k(kernel_size)
        return padding

    def fill_weights_NN(self,r:int,
                             s:int)->None:
        kernel_size = s
        padding = self.compute_padding(kernel_size)
        conv3d = torch.nn.Conv3d(in_channels=1, \
                                      out_channels=r**2, \
                                      kernel_size=[1, kernel_size, kernel_size], \
                                      stride=[1,s,s], \
                                      padding=[0,padding,padding], \
                                      padding_mode='replicate',
                                      bias=False)
        new_weights = np.zeros(conv3d.weight.shape)
        a = s/r
        nearest_neighbours = np.round(np.linspace(a/2,(s-1)-(a/2),r)).astype(int)
        nx, ny = np.meshgrid(nearest_neighbours,nearest_neighbours)
        for r_i, (y, x) in enumerate(zip(nx.flatten(), ny.flatten())):
            new_weights[r_i,0, 0, x, y] = 1.
        conv3d.weight = torch.nn.Parameter(torch.Tensor(new_weights))
        return conv3d

    def fill_weights_BiLinear(self,r:int,
                        s:int)->None:
        kernel_size = s
        padding = self.compute_padding(kernel_size)
        conv3d = torch.nn.Conv3d(in_channels=1, \
                                 out_channels=r**2, \
                                 kernel_size=[1, kernel_size, kernel_size], \
                                 stride=[1,s,s], \
                                 padding=[0,padding,padding], \
                                 padding_mode='replicate',
                                 bias=False)
        new_weights = np.zeros(conv3d.weight.shape)
        a = s/r
        nearest_neighbours = np.linspace(a/2,(s-1)-(a/2),r)
        nx, ny = np.meshgrid(nearest_neighbours,nearest_neighbours)
        for r_i, (y, x) in enumerate(zip(nx.flatten(), ny.flatten())):
            x_0, x_1 = np.floor(x).astype(int), np.ceil(x).astype(int)
            y_0, y_1 = np.floor(y).astype(int), np.ceil(y).astype(int)
            if x_1 == x_0:
                x_1 += 1
            if y_1 == y_0:
                y_1 += 1

            new_weights[r_i,0, 0, x_0, y_0] = (x_1-x)*(y_1-y)
            if y_1 < kernel_size:
                new_weights[r_i,0, 0, x_0, y_1] = (x_1-x)*(y-y_0)
            if x_1 < kernel_size:
                new_weights[r_i,0, 0, x_1, y_0] = (x-x_0)*(y_1-y)
            if (x_1 < kernel_size) and (y_1 < kernel_size):
                new_weights[r_i,0, 0, x_1, y_1] = (x-x_0)*(y-y_0)
        conv3d.weight = torch.nn.Parameter(torch.Tensor(new_weights))
        return conv3d
    def fill_weights_BiQubic(self,r:int,
                              s:int)->None:
        kernel_size = s+1
        padding = self.compute_padding(kernel_size)
        conv3d = torch.nn.Conv3d(in_channels=1, \
                                 out_channels=r**2, \
                                 kernel_size=[1, kernel_size, kernel_size], \
                                 stride=[1,s,s], \
                                 padding=[0,padding,padding], \
                                 padding_mode='replicate',
                                 bias=False)
        new_weights = np.zeros(conv3d.weight.shape)
        a = s/r
        nearest_neighbours = np.linspace(a/2,(s-1)-(a/2),r)
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
                new_weights[r_i,0, 0, x_0:x_3+1, y_0:y_3+1] = W[0:x_3-x_0+1,0:y_3-y_0+1] / ((W[0:x_3-x_0+1,0:y_3-y_0+1]).sum() if (W[0:x_3-x_0+1,0:y_3-y_0+1]).sum() > 0 else 1)
            elif (x_3 < kernel_size) and (0<=x_0) and (y_3 < kernel_size+1) and (0 <= y_0):
                new_weights[r_i,0, 0, x_0:x_3+1, y_0:y_2+1] = W[:,y_0:y_2+1]/(W[:,y_0:y_2+1]).sum()
            elif (x_3 < kernel_size+1) and (0<=x_0) and (y_3 < kernel_size) and (0 < y_0):
                new_weights[r_i,0, 0, x_0:x_2+1, y_0:y_3+1] = W[x_0:x_2+1,:]/(W[x_0:x_2+1,:]).sum()
            elif (x_3 < kernel_size+1) and (0<=x_0) and (y_3 < kernel_size+1) and (0 <= y_0):
                new_weights[r_i,0, 0, x_0:x_2+1, y_0:y_2+1] = W[x_0:x_2+1,y_0:y_2+1]/(W[x_0:x_2+1,y_0:y_2+1]).sum()
            elif (x_3 < kernel_size) and (0<=x_0) and (y_3 < kernel_size+2) and (0 <= y_0):
                new_weights[r_i,0, 0, x_0:x_3+1, y_0:y_1+1] = W[:,y_0:y_1+1]/(W[:,y_0:y_1+1]).sum()
            elif (x_3 < kernel_size+2) and (0<=x_0) and (y_3 < kernel_size) and (0 <=y_0):
                new_weights[r_i,0, 0, x_0:x_1+1, y_0:y_3+1] = W[x_0:x_1+1,:]/(W[x_0:x_1+1,:]).sum()
            elif (x_3 < kernel_size+1) and (0<=x_0) and (y_3 < kernel_size+2) and (0 <= y_0):
                new_weights[r_i,0, 0, x_0:x_2+1, y_0:y_1+1] = W[x_0:x_2+1,y_0:y_1+1]/(W[x_0:x_2+1,y_0:y_1+1]).sum()
            elif (x_3 < kernel_size+2) and (0<=x_0) and (y_3 < kernel_size+1) and (0 <=y_0):
                new_weights[r_i,0, 0, x_0:x_1+1, y_0:y_2+1] = W[x_0:x_1+1,y_0:y_2+1]/(W[x_0:x_1+1,y_0:y_2+1]).sum()
            elif (x_3 < kernel_size+2) and (0<=x_0) and (y_3 < kernel_size+2) and (0 <= y_0):
                new_weights[r_i,0, 0, x_0:x_1+1, y_0:y_1+1] = W[x_0:x_1+1,y_0:y_1+1]/(W[x_0:x_1+1,y_0:y_1+1]).sum()
            else:
                new_weights[r_i,0, 0, x_1, y_1]=1
        conv3d.weight = torch.nn.Parameter(torch.Tensor(new_weights))
        return conv3d

    def __init__(self,
                 r:             int,
                 s:             int,
                 scaling_mode: str='nearest') -> None:
        super(FullyConvolutionalFractionalScaling2D, self).__init__()
        self.scaling_modes = {
            'bicubic':  self.fill_weights_BiQubic,
            'nearest':  self.fill_weights_NN,
            'bilinear': self.fill_weights_BiLinear
        }
        self.conv3d = self.scaling_modes[scaling_mode](r, s).to("cuda:0") #self.fill_weights_BiLInear(r, s)
        self.pixelshuffle = torch.nn.PixelShuffle(upscale_factor=r)
        self(torch.Tensor(np.zeros((1,1024,1024,3))).to('cuda:0'))

    def forward(self,input: torch.Tensor) -> torch.Tensor:
        reduce_dim = False
        if len(input.shape) == 3:
            input = input[None, ...]
            reduce_dim = True
        x = torch.permute(input, (0,3,1,2))
        x = x[:, None, :, :, :]
        x = self.conv3d(x)
        x = torch.permute(x, (0,2,1,3,4))
        x = self.pixelshuffle(x)
        x = torch.squeeze(x, 2)
        res = torch.permute(x, (0,2,3,1))
        if reduce_dim:
            res = res[0]
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
#     torch0 = torchvision.transforms.Resize(size=(5221,5224), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
#     torch1 = torchvision.transforms.Resize(size=(5221,5224), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
#     torch2 = torchvision.transforms.Resize(size=(5221,5224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
#     times.append(time.time())
#     B = FCFS0(torch.Tensor(A))
#     # print(B.shape)
#     # plt.figure(0)
#     # plt.imshow(B.detach().numpy().astype(int))
#
#     times.append(time.time())
#     B2 = FCFS1(torch.Tensor(A))
#     # print(B2.shape)
#     # plt.figure(1)
#     # plt.imshow(B2.detach().numpy().astype(int))
#
#     times.append(time.time())
#     B2 = FCFS2(torch.Tensor(A))
#     # print(B2.shape)
#     # plt.figure(2)
#     # plt.imshow(B2.detach().numpy().astype(int))
#
#     times.append(time.time())
#     B2 = torch0(torch.Tensor(A).permute([2, 0, 1]))
#     # print(B2.shape)
#     # plt.figure(3)
#     # plt.imshow(B2.permute([1, 2, 0]).detach().numpy().astype(int))
#
#     times.append(time.time())
#     B2 = torch1(torch.Tensor(A).permute([2, 0, 1]))
#     # print(B2.shape)
#     # plt.figure(4)
#     # plt.imshow(B2.permute([1, 2, 0]).detach().numpy().astype(int))
#
#     times.append(time.time())
#     B2 = torch2(torch.Tensor(A).permute([2, 0, 1]))
#     # print(B2.shape)
#     # plt.figure(5)
#     # plt.imshow(B2.permute([1, 2, 0]).detach().numpy().astype(int))
#     times.append(time.time())
#     for t1,t2 in zip(times[:-1],times[1:]):
#         print(t2 - t1)
#     plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
