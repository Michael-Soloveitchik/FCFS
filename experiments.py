import os
import timeit
import numpy.random
from tqdm import tqdm
import cv2
import numpy as np
from fully_convolutional_fractional_scale_layer import *
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

import torchmetrics
def set_grid(fig):
    ax = fig.add_subplot(1, 1, 1)
    # Major ticks every 20, minor ticks every 5
    ax.xaxis.set_major_locator(MultipleLocator(15))
    ax.yaxis.set_major_locator(MultipleLocator(15))

    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.7)
    return ax

interpulation_modes = [('nearest',torchvision.transforms.InterpolationMode.NEAREST), \
                       ('bilinear',torchvision.transforms.InterpolationMode.BILINEAR), \
                       ('bicubic',torchvision.transforms.InterpolationMode.BICUBIC)
                       ]
models = [FullyConvolutionalFractionalScaling2D, \
          torchvision.transforms.Resize
          ]
#MACROS
TIMES = 20
STAGES = 11
START = 3
blank = np.ones((900,200,3))*255.
space = np.ones((300, 900*3+200*3,3))*255.
small_space_1 = np.ones((52, 2048,3))*255.
small_space_2 = np.ones((2400-2100, 2048,3))*255.

# import torchmetrics
psnr = torchmetrics.PeakSignalNoiseRatio()
ssim = torchmetrics.StructuralSimilarityIndexMeasure()
#DOWN-SCALING
# HQ path
HQ_1024_CELEBA_path = r"C:\Users\micha\Documents\data512x512"
random_files = numpy.random.permutation(len(os.listdir(HQ_1024_CELEBA_path)))


if 1 and 1:
    # initializing notebook
    our_notebook_downscaling = np.zeros((TIMES,STAGES-2,3),dtype=np.float64)
    our_list_downscaling = [[None],
                            [None],
                            [None]]

    torch_notebook_downscaling = np.zeros((TIMES,STAGES-2,3),dtype=np.float64)
    torch_list_downscaling = [[None],
                              [None],
                              [None]]

    psnr_notebook_downscaling = np.zeros((TIMES,STAGES-2,3))
    ssim_notebook_downscaling = np.zeros((TIMES,STAGES-2,3))
    # repeating 20 times
    for t in tqdm(range(TIMES)):
        # load random image 1024x1024
        file = os.listdir(HQ_1024_CELEBA_path)[random_files[t]]
        input_im = cv2.imread(os.path.join(HQ_1024_CELEBA_path, file))
        for inter_i, interpulation_mode in enumerate(interpulation_modes):
            for stage_i, ratio in enumerate(np.arange(STAGES, 2, -1)):

                # OURS!
                # Be ready!
                our_model = FullyConvolutionalFractionalScaling2D(r=ratio, s=STAGES, scaling_mode=interpulation_mode[0])
                # Start!
                our_input_im = torch.Tensor(input_im).to('cuda:0')
                our_model((torch.Tensor(input_im*0.)).to('cuda:0'))
                # GO!
                t1 =timeit.default_timer()
                our_output_im = our_model(our_input_im)
                t2 = timeit.default_timer()
                our_notebook_downscaling[t,stage_i,inter_i] = t2-t1
                our_list_downscaling[inter_i][0] = ([cv2.cvtColor(input_im,cv2.COLOR_BGR2RGB), cv2.cvtColor(our_output_im.detach().cpu().numpy(),cv2.COLOR_BGR2RGB).astype(int)])

                # Torch!
                # Be ready!
                torch_model = torchvision.transforms.Resize(size=tuple(np.array(our_output_im.shape[:2])), interpolation=interpulation_mode[1])
                # Start!
                torch_input_im = torch.Tensor(input_im).permute([2, 0, 1]).to('cuda:0')
                torch_model(torch.Tensor(input_im*0.).permute([2, 0, 1]).to('cuda:0'))
                # GO!
                t1 = timeit.default_timer()
                torch_output_im = torch_model(torch_input_im).permute(1,2,0)
                t2 = timeit.default_timer()
                torch_notebook_downscaling[t,stage_i,inter_i] = t2-t1
                torch_list_downscaling[inter_i][0] = ([cv2.cvtColor(input_im,cv2.COLOR_BGR2RGB), cv2.cvtColor(torch_output_im.detach().cpu().numpy(),cv2.COLOR_BGR2RGB).astype(int)])
                psnr_notebook_downscaling[t,stage_i,inter_i] = psnr(torch_output_im.detach().cpu().permute(2,0,1)[None,...],our_output_im.detach().cpu().permute(2,0,1)[None,...])
                ssim_notebook_downscaling[t,stage_i,inter_i] = ssim(torch_output_im.detach().cpu().permute(2,0,1)[None,...],our_output_im.detach().cpu().permute(2,0,1)[None,...])
    psnr_notebook_downscaling[np.isinf(psnr_notebook_downscaling)]=100

    # Presentation
    dwn_img = [
        [blank, [],blank,[],blank,[]],
        space,
        [blank, [],blank,[],blank,[]],
        space
    ]
    for i in range(3):
        fig=plt.figure(i+22,figsize=(10, 7))
        set_grid(fig)
        plt.title(f'Downscaling Efficency\n Interpulation mode - "{interpulation_modes[i][0]}"')
        plt.errorbar(x=np.arange(STAGES-2),y=torch_notebook_downscaling.mean(0)[:,i], yerr=torch_notebook_downscaling.std(0)[:,i], label='Torch')
        plt.errorbar(x=np.arange(STAGES-2),y=our_notebook_downscaling.mean(0)[:,i], yerr=our_notebook_downscaling.std(0)[:,i], label='FCFS')
        plt.xlabel('down-scaling ratio')
        plt.xticks(np.arange(STAGES-2), [str(a) for a in np.arange(STAGES,2,-1)/STAGES])
        plt.yticks([a for a in np.concatenate([torch_notebook_downscaling.mean(0)[:,i], our_notebook_downscaling.mean(0)[:,i]], axis=0)])
        plt.legend()
        plt.savefig(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\Down-scaling - {interpulation_modes[i][0]} - Efficiency.jpg")

        fig=plt.figure(i*i+33,figsize=(10, 7))
        set_grid(fig)
        plt.title(f'Downscaling PSNR\n Interpulation mode - "{interpulation_modes[i][0]}"')
        plt.errorbar(x=np.arange(STAGES-2),y=psnr_notebook_downscaling.mean(0)[:,i], yerr=psnr_notebook_downscaling.std(0)[:,i], label='psnr')
        plt.xlabel('Downscaling ratio')
        plt.xticks(np.arange(STAGES-2), [str(a) for a in np.arange(STAGES,2,-1)/STAGES])
        plt.yticks([a for a in psnr_notebook_downscaling.mean(0)[:,i]])
        plt.legend()
        # plt.show()
        plt.savefig(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\Down-scaling - {interpulation_modes[i][0]} - PSNR.jpg")
        print(f'Down-scaling PSNR\n Interpulation mode - "{interpulation_modes[i][0]}"')
        print(psnr_notebook_downscaling.mean(0)[-1,i], "+/-",psnr_notebook_downscaling.std(0)[-1,i])

        fig=plt.figure(i*i+44,figsize=(10, 7))
        fig.clear()
        set_grid(fig)
        plt.title(f'Down-scaling SSIM\n Interpulation mode - "{interpulation_modes[i][0]}"')
        plt.errorbar(x=np.arange(STAGES-2),y=ssim_notebook_downscaling.mean(0)[:,i],yerr=ssim_notebook_downscaling.std(0)[:,i], label='SSIM')
        plt.xlabel('down-scaling ratio')
        plt.xticks(np.arange(STAGES-2), [str(a) for a in np.arange(STAGES,2,-1)/STAGES])
        plt.yticks([a for a in ssim_notebook_downscaling.mean(0)[:,i]])
        plt.legend()
        plt.savefig(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\Down-scaling - {interpulation_modes[i][0]} - SSIM.jpg")

        plt.legend()
        fig, axes = plt.subplots(2,2,figsize=(15,15))
        fig.canvas.set_window_title(f"Down-scaling ({interpulation_modes[i][0]})")
        x= np.copy(our_list_downscaling[i][0][0])
        cv2.imwrite(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\FCFS Input - Down-scaling({interpulation_modes[i][0]}).jpg", cv2.cvtColor(np.repeat(np.repeat(x, int((3000//x.shape[0])+1), axis=0), int((3000//x.shape[1])+1), axis=1).astype(np.float32), cv2.COLOR_RGB2BGR))
        axes[0,0].imshow(x)
        axes[0,0].set_title('FCFS Input')

        x = np.copy(our_list_downscaling[i][0][1]*1.0)
        cv2.imwrite(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\FCFS Output - Down-scaling({interpulation_modes[i][0]}).jpg", cv2.cvtColor(np.repeat(np.repeat(x, int((3000//x.shape[0])+1), axis=0), int((3000//x.shape[1])+1), axis=1).astype(np.float32)[1050:1950,1050:1950], cv2.COLOR_RGB2BGR))
        axes[0,1].imshow(x)
        axes[0,1].set_title('FCFS Output')
        dwn_img[0][1+i+i] = np.copy(cv2.cvtColor(np.repeat(np.repeat(x, int((3000//x.shape[0])+1), axis=0), int((3000//x.shape[1])+1), axis=1).astype(np.float32)[1050:1950,1050:1950], cv2.COLOR_RGB2BGR)*1.)

        x = np.copy(torch_list_downscaling[i][0][0])
        input = np.copy(cv2.cvtColor(np.repeat(np.repeat(x, int((2000//x.shape[0])+1), axis=0), int((2000//x.shape[1])+1), axis=1).astype(np.float32), cv2.COLOR_RGB2BGR)*1.)
        cv2.imwrite(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\Torch Input - Down-scaling({interpulation_modes[i][0]}).jpg", cv2.cvtColor(np.repeat(np.repeat(x, int((3000//x.shape[0])+1), axis=0), int((3000//x.shape[1])+1), axis=1).astype(np.float32), cv2.COLOR_RGB2BGR))
        axes[1,0].imshow(x)
        axes[1,0].set_title('Torch Input')

        x = np.copy(torch_list_downscaling[i][0][1])
        cv2.imwrite(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\Torch Output - Down-scaling({interpulation_modes[i][0]}).jpg", cv2.cvtColor(np.repeat(np.repeat(x, int((3000//x.shape[0])+1), axis=0), int((3000//x.shape[1])+1), axis=1).astype(np.float32)[1050:1950,1050:1950], cv2.COLOR_RGB2BGR))
        axes[1,1].imshow(x)
        axes[1,1].set_title('Torch Output')
        dwn_img[2][1+i+i] = np.copy(cv2.cvtColor(np.repeat(np.repeat(x, int((3000//x.shape[0])+1), axis=0), int((3000//x.shape[1])+1), axis=1).astype(np.float32)[1050:1950,1050:1950], cv2.COLOR_RGB2BGR)*1.)
        # plt.show()
    dwn_img[0]=np.concatenate(dwn_img[0], axis=1)
    dwn_img[2]=np.concatenate(dwn_img[2], axis=1)
    dwn_img = np.concatenate(dwn_img, axis=0)
    dwn_img = np.concatenate([np.concatenate([small_space_1,input,small_space_2],axis=0),dwn_img],axis=1)
    cv2.imwrite(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\Down-scaling-Matrix.jpg", dwn_img)

# UPSCALING
if 1 and 0:
    our_notebook_upscaling = np.zeros((TIMES,STAGES-2,3),dtype=np.float64)
    our_list_upscaling = [[None],
                            [None],
                            [None]]

    torch_notebook_upscaling = np.zeros((TIMES,STAGES-2,3),dtype=np.float64)
    torch_list_upscaling = [[None],
                              [None],
                              [None]]

    psnr_notebook_upscaling = np.zeros((TIMES,STAGES-2,3))
    ssim_notebook_upscaling = np.zeros((TIMES,STAGES-2,3))
    # repeating 20 times
    for t in tqdm(range(TIMES)):
        # load random image 1024x1024
        file = os.listdir(HQ_1024_CELEBA_path)[random_files[t]]
        input_im = cv2.imread(os.path.join(HQ_1024_CELEBA_path, file))
        for inter_i, interpulation_mode in enumerate(interpulation_modes):
            for stage_i, ratio in enumerate(np.arange(0, (3*(STAGES-2)), 3)):

                # OURS!
                # Be ready!
                our_model = FullyConvolutionalFractionalScaling2D(r=STAGES+ratio, s=STAGES, scaling_mode=interpulation_mode[0])
                # Start!
                our_input_im = torch.Tensor(input_im).to('cuda:0')
                our_model((torch.Tensor(input_im*0.)).to('cuda:0'))
                # GO!
                t1 = timeit.default_timer()
                our_output_im = our_model(our_input_im)
                t2 = timeit.default_timer()
                our_notebook_upscaling[t,stage_i,inter_i] = t2-t1
                our_list_upscaling[inter_i][0] = ([cv2.cvtColor(input_im,cv2.COLOR_BGR2RGB), cv2.cvtColor(our_output_im.detach().cpu().numpy(),cv2.COLOR_BGR2RGB).astype(int)])

                # Torch!
                # Be ready!
                torch_model = torchvision.transforms.Resize(size=tuple(np.array(our_output_im.shape[:2])), interpolation=interpulation_mode[1])
                # Start!
                torch_input_im = torch.Tensor(input_im).permute(2, 0, 1).to('cuda:0')
                torch_model(torch.Tensor(input_im*0.).permute([2, 0, 1]).to('cuda:0'))
            # GO!
                t1 = timeit.default_timer()
                torch_output_im = torch_model(torch_input_im).permute(1,2,0)
                t2 = timeit.default_timer()
                torch_notebook_upscaling[t,stage_i,inter_i] = t2 -t1
                torch_list_upscaling[inter_i][0] = ([cv2.cvtColor(input_im,cv2.COLOR_BGR2RGB), cv2.cvtColor(torch_output_im.detach().cpu().numpy(),cv2.COLOR_BGR2RGB).astype(int)])

                psnr_notebook_upscaling[t,stage_i,inter_i] = psnr(torch_output_im.detach().cpu().permute(2,0,1)[None,...],our_output_im.detach().cpu().permute(2,0,1)[None,...])
                ssim_notebook_upscaling[t,stage_i,inter_i] = ssim(torch_output_im.detach().cpu().permute(2,0,1)[None,...],our_output_im.detach().cpu().permute(2,0,1)[None,...])
    psnr_notebook_upscaling[np.isinf(psnr_notebook_upscaling)]=100
    # Presentation
    up_img = [
        [blank, [],blank,[],blank,[]],
        space,
        [blank, [],blank,[],blank,[]],
        space,
    ]
    print('UP-SACLING')
    for i in range(3):
        fig=plt.figure(i*i+22,figsize=(10, 7))
        fig.clear()
        set_grid(fig)
        plt.title(f'Up-scaling eficency\n Interpulation mode - "{interpulation_modes[i][0]}"')
        plt.errorbar(x=np.arange(STAGES-2),y=torch_notebook_upscaling.mean(0)[:,i], yerr=torch_notebook_upscaling.std(0)[:,i], label='Torch')
        plt.errorbar(x=np.arange(STAGES-2),y=our_notebook_upscaling.mean(0)[:,i], yerr=our_notebook_upscaling.std(0)[:,i], label='FCFS')
        plt.xlabel('up-scaling ratio')
        plt.xticks(np.arange(STAGES-2))
        plt.yticks([a for a in np.concatenate([torch_notebook_upscaling.mean(0)[:,i], our_notebook_upscaling.mean(0)[:,i]], axis=0)])
        plt.legend()
        plt.savefig(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\Up-scaling - {interpulation_modes[i][0]} - Efficiency.jpg")

        fig=plt.figure(i*i+33,figsize=(10, 7))
        fig.clear()
        set_grid(fig)
        plt.title(f'Up-scaling PSNR\n Interpulation mode - "{interpulation_modes[i][0]}"')
        plt.errorbar(x=np.arange(STAGES-2),y=psnr_notebook_upscaling.mean(0)[:,i], yerr=psnr_notebook_upscaling.std(0)[:,i], label='PSNR')
        plt.xlabel('up-scaling ratio')
        plt.xticks(np.arange(STAGES-2))
        plt.yticks([a for a in psnr_notebook_upscaling.mean(0)[:,i]])
        plt.legend()
        plt.savefig(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\Up-scaling - {interpulation_modes[i][0]} - PSNR.jpg")
        print(f'Up-scaling PSNR\n Interpulation mode - "{interpulation_modes[i][0]}"')
        print(psnr_notebook_upscaling.mean(0)[-1,i], '+/-',psnr_notebook_upscaling.std(0)[-1,i])

        fig=plt.figure(i*i+44,figsize=(10, 7))
        fig.clear()
        set_grid(fig)
        plt.title(f'Up-scaling SSIM\n Interpulation mode - "{interpulation_modes[i][0]}"')
        plt.errorbar(x=np.arange(STAGES-2),y=ssim_notebook_upscaling.mean(0)[:,i],yerr=ssim_notebook_upscaling.std(0)[:,i], label='SSIM')
        plt.xlabel('up-scaling ratio')
        plt.xticks(np.arange(STAGES-2))
        plt.yticks([a for a in ssim_notebook_upscaling.mean(0)[:,i]])
        plt.legend()
        plt.savefig(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\Up-scaling - {interpulation_modes[i][0]} - SSIM.jpg")

        print(f'Up-scaling SSIM\n Interpulation mode - "{interpulation_modes[i][0]}"')
        print(ssim_notebook_upscaling.mean(0)[0,i], '+/-', ssim_notebook_upscaling.std(0)[0,i])
        fig, axes = plt.subplots(2,2,figsize=(15,15))
        fig.canvas.set_window_title(f"UP-scaling ({interpulation_modes[i][0]})")
        x = np.copy(our_list_upscaling[i][-1][0])
        cv2.imwrite(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\FCFS Output - Up-scaling({interpulation_modes[i][0]}).jpg", cv2.cvtColor(np.repeat(np.repeat(x, int((3000//x.shape[0])+1), axis=0), int((3000//x.shape[1])+1), axis=1).astype(np.float32), cv2.COLOR_RGB2BGR))
        axes[0,0].imshow(x)
        axes[1,1].set_title('FCFS Input')

        x = np.copy(our_list_upscaling[i][-1][1])
        cv2.imwrite(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\FCFS Output - Up-scaling({interpulation_modes[i][0]}).jpg", cv2.cvtColor(np.repeat(np.repeat(x, int((3000//x.shape[0])+1), axis=0), int((3000//x.shape[1])+1), axis=1).astype(np.float32)[1550:2450,1550:2450], cv2.COLOR_RGB2BGR))
        axes[0,1].imshow(x)
        axes[1,1].set_title('FCFS Input')
        up_img[0][1+i+i] = np.copy(cv2.cvtColor(np.repeat(np.repeat(x, int((3000//x.shape[0])+1), axis=0), int((3000//x.shape[1])+1), axis=1).astype(np.float32)[1550:2450,1550:2450], cv2.COLOR_RGB2BGR)*1.)

        x = np.copy(torch_list_upscaling[i][-1][0])
        input = np.copy(cv2.cvtColor(np.repeat(np.repeat(x, int((2000//x.shape[0])+1), axis=0), int((2000//x.shape[1])+1), axis=1).astype(np.float32), cv2.COLOR_RGB2BGR)*1.)
        cv2.imwrite(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\Torch Input - Up-scaling({interpulation_modes[i][0]}).jpg", cv2.cvtColor(np.repeat(np.repeat(x, int((3000//x.shape[0])+1), axis=0), int((3000//x.shape[1])+1), axis=1).astype(np.float32), cv2.COLOR_RGB2BGR))
        axes[1,0].imshow(x)
        axes[1,0].set_title('Torch Input')

        x = np.copy(torch_list_upscaling[i][-1][1])
        cv2.imwrite(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\Torch Output - Up-scaling({interpulation_modes[i][0]}).jpg", cv2.cvtColor(np.repeat(np.repeat(x, int((3000//x.shape[0])+1), axis=0), int((3000//x.shape[1])+1), axis=1).astype(np.float32)[1550:2450,1550:2450], cv2.COLOR_RGB2BGR))
        axes[1,1].imshow(x)
        axes[1,1].set_title('Torch Output')
        up_img[2][1+i+i] = np.copy(cv2.cvtColor(np.repeat(np.repeat(x, int((3000//x.shape[0])+1), axis=0), int((3000//x.shape[1])+1), axis=1).astype(np.float32)[1550:2450,1550:2450], cv2.COLOR_RGB2BGR)*1.)
        # plt.show()
    up_img[0]=np.concatenate(up_img[0], axis=1)
    up_img[2]=np.concatenate(up_img[2], axis=1)
    up_img = np.concatenate(up_img, axis=0)
    up_img = np.concatenate([np.concatenate([small_space_1,input,small_space_2],axis=0),up_img],axis=1)
    cv2.imwrite(r"C:\\Users\\micha\\Pictures\\FCFS1"+f"\\Up-scaling-Matrix.jpg", up_img)
# del our_list_upscaling
# del torch_list_upscaling
# our_list_upscaling= [[],[],[]]
# torch_list_upscaling = [[],[],[]]
# looping 10 stages

#
# #UPSCALING
# FCFS_models = []
# Torch_models = []


# FCFS0 = FullyConvolutionalFractionalScaling2D(r=23,s=5,scaling_mode='nearest') # downsampling by factor 2/3
# FCFS1 = FullyConvolutionalFractionalScaling2D(r=23,s=5,scaling_mode='bilinear') # downsampling by factor 2/3
# FCFS2 = FullyConvolutionalFractionalScaling2D(r=23,s=3,scaling_mode='bicubic') # downsampling by factor 2/3
# torch0 = torchvision.transforms.Resize(size=(5221,5224), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
# torch1 = torchvision.transforms.Resize(size=(5221,5224), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
# torch2 = torchvision.transforms.Resize(size=(5221,5224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
# times.append(timeit.default_timer())
# B = FCFS0(torch.Tensor(A))
# # print(B.shape)
# # plt.figure(0)
# # plt.imshow(B.detach().numpy().astype(int))
#
# times.append(timeit.default_timer())
# B2 = FCFS1(torch.Tensor(A))
# # print(B2.shape)
# # plt.figure(1)
# # plt.imshow(B2.detach().numpy().astype(int))
#
# times.append(timeit.default_timer())
# B2 = FCFS2(torch.Tensor(A))
# # print(B2.shape)
# # plt.figure(2)
# # plt.imshow(B2.detach().numpy().astype(int))
#
# times.append(timeit.default_timer())
# B2 = torch0(torch.Tensor(A).permute([2, 0, 1]))
# # print(B2.shape)
# # plt.figure(3)
# # plt.imshow(B2.permute([1, 2, 0]).detach().numpy().astype(int))
#
# times.append(timeit.default_timer())
# B2 = torch1(torch.Tensor(A).permute([2, 0, 1]))
# # print(B2.shape)
# # plt.figure(4)
# # plt.imshow(B2.permute([1, 2, 0]).detach().numpy().astype(int))
#
# times.append(timeit.default_timer())
# B2 = torch2(torch.Tensor(A).permute([2, 0, 1]))
# # print(B2.shape)
# # plt.figure(5)
# # plt.imshow(B2.permute([1, 2, 0]).detach().numpy().astype(int))
# times.append(timeit.default_timer())
# for t1,t2 in zip(times[:-1],times[1:]):
#     print(t2 - t1)
# plt.show()