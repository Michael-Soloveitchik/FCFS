# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torchvision
from typing import Optional, List, Tuple, Union


class FullyConvolutionalFractionalScaling2D(torch.nn.Module):
    def fill_weights_NN(self):
        pass
    def __init__(self,
                 r: Tuple[int, ...],
                 s: Tuple[int, ...],
                 in_channels: int,
                 kernel_size: Tuple[int, ...],
                 padding_mode: str='repeat',
                 scalling_mode: str=None) -> None:
        super(FullyConvolutionalFractionalScaling2D, self).__init__()
        self.scaling_modes = {
            'bicubic':  None,
            'nearest':  None,
            'bilinear': None
        }

        compute_padding_from_k = lambda x: x//2-1 if (x%2==0) else x//2
        if torch.is_tensor(kernel_size):
            kernel_size = kernel_size.numpy()
        padding = [compute_padding_from_k(k) for k in kernel_size] if isinstance(kernel_size, (list, tuple, np.ndarray)) else compute_padding_from_k(kernel_size)

        self.conv2d = torch.nn.Conv2d(in_channels=in_channels, out_channels=r**2, kernel_size=kernel_size, stride=s, padding=padding,padding_mode=padding_mode)#(kernel_size//2-1))
        self.pixelshuffle = torch.nn.PixelShuffle(upscale_factor=r)

    def forward(self,input: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(input)
        res = self.pixelshuffle(x)
        return res


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import numpy as np
    FCFS = FullyConvolutionalFractionalScaling2D(r=2,s=3,in_channels=1,kernel_size=[3,2], padding_mode) # downsampling by factor 2/3
    A = np.ones((1,1,17,17))
    B = FCFS(torch.Tensor(A))
    print(B.shape)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
