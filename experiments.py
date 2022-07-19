import os
import timeit
import numpy.random
from tqdm import tqdm
import cv2
import numpy as np
from fully_convolutional_fractional_scale_layer import *
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

import torchmetrics
def set_grid(fig):
    ax = fig.add_subplot(1, 1, 1)
    # Major ticks every 20, minor ticks every 5
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    # Change minor ticks to show every 5. (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(0.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(0.1))
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.7)
    return ax

# MACROS
# Final poster
REPEAT_EXPERIMENT_TIMES = 15
FRACTIONS = 11
START=3
PAD_MULTIPLYER=5
device = 'cuda:0'
down_window = (150, 220)
up_window = (390, 590)
print(device)
# Models & Algos
interpulation_modes = [(cv2.INTER_NEAREST,torchvision.transforms.InterpolationMode.NEAREST), \
                       (cv2.INTER_LINEAR,torchvision.transforms.InterpolationMode.BILINEAR), \
                       (cv2.INTER_CUBIC,torchvision.transforms.InterpolationMode.BICUBIC)
                       ]
models = [FullyConvolutionalFractionalScaling2D, \
          torchvision.transforms.Resize
          ]

# import torchmetrics
psnr = torchmetrics.PeakSignalNoiseRatio()
ssim = torchmetrics.StructuralSimilarityIndexMeasure()

#DOWN-SCALING
# HQ path
HQ_1024_CELEBA_path = r"/content/drive/MyDrive/HUJI/Michael Werman/Datasets/data512x512"
dump_path = r"/content/drive/MyDrive/HUJI/Michael Werman/FCFS"
# HQ_1024_CELEBA_path = r"/mnt/g/My Drive/HUJI/Michael Werman/Datasets/data512x512"
# dump_path = r"/mnt/g/My Drive/HUJI/Michael Werman/FCFS"

random_files = numpy.random.permutation(len(os.listdir(HQ_1024_CELEBA_path)))


def run_rescaling_experiment_on_range(fractions, dump_pictures_on_fraction,window, down=False):
    def warm_up_gpu(model,shape):
        for i in range(100):
            _ = model(torch.randn(shape).to(device))
        # print("Done Worming-Up")
        return model, model(torch.randn(shape).to(device)).shape
    def initialize_result_images_and_accomulation_metrics():
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), np.zeros((REPEAT_EXPERIMENT_TIMES, FRACTIONS-START, 3),dtype=np.float64)
    def padd_image(image,multiplyer=PAD_MULTIPLYER):
        return np.repeat(np.repeat(image,multiplyer,0),multiplyer,1)
    FCFS_starter, FCFS_ender, FCFS_time_diffs = initialize_result_images_and_accomulation_metrics()
    torch_starter1, torch_ender1, torch_time_diffs = initialize_result_images_and_accomulation_metrics()
    torch_starter2, torch_ender2, _ = initialize_result_images_and_accomulation_metrics()
    psnr_notebook_rescaling = np.zeros((REPEAT_EXPERIMENT_TIMES, FRACTIONS-START, 3))
    ssim_notebook_rescaling = np.zeros((REPEAT_EXPERIMENT_TIMES, FRACTIONS-START, 3))

    poster = None
    # repeating MULTIPLYERS REPEAT_TIMES
    for interpulation_mode_i, interpulation_mode in enumerate(interpulation_modes):
        # load random image 1024x1024
        file_name = os.listdir(HQ_1024_CELEBA_path)[random_files[0]]
        input_im = np.transpose(cv2.imread(os.path.join(HQ_1024_CELEBA_path, file_name)), (2,0,1))[None,...]
        input_im_cuda = torch.Tensor(input_im).to(device)
        # print(len(fractions))
        for fraction_i, (r, s) in tqdm(enumerate(fractions)):
            # Initialize FCFS the model
            FCFS_model = FullyConvolutionalFractionalScaling2D(r=r, s=s, scaling_mode=interpulation_mode[0], is_inner_layer=True, device=device)
            # Warm-UP!
            FCFS_model, FCFS_shape = warm_up_gpu(FCFS_model, input_im_cuda.shape)
            with torch.no_grad():
                for t in range(REPEAT_EXPERIMENT_TIMES):
                    _ = FCFS_model(input_im_cuda)
                    FCFS_starter.record()
                    _ = FCFS_model(input_im_cuda)
                    FCFS_ender.record()
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    curr_time = FCFS_starter.elapsed_time(FCFS_ender)
                    FCFS_time_diffs[t,fraction_i,interpulation_mode_i] = curr_time
                    # print(curr_time)

            # Torch model initialzation
            with torch.no_grad():
                for t in range(REPEAT_EXPERIMENT_TIMES):
                    # _ = torch_model(input_im_cuda)
                    torch_starter1.record()
                    torch_model = torchvision.transforms.Resize(size=tuple(np.array(FCFS_shape[-2:])),
                                                                interpolation=interpulation_mode[1])
                    torch_ender1.record()
                    torch.cuda.synchronize()
                    curr_time1 = torch_starter1.elapsed_time(torch_ender1)

                    torch_model, _ = warm_up_gpu(torch_model, input_im_cuda.shape)

                    torch_starter2.record()
                    _ = torch_model(input_im_cuda)
                    torch_ender2.record()
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    curr_time2 = torch_starter2.elapsed_time(torch_ender2)
                    torch_time_diffs[t, fraction_i, interpulation_mode_i] = curr_time1+curr_time2
                    # print(curr_time)
            # PSNR & SSIM metrics
            for t in range(REPEAT_EXPERIMENT_TIMES):
                FCFS_output_im = FCFS_model(input_im_cuda).detach().cpu()
                torch_output_im = torch_model(input_im_cuda).detach().cpu()

                psnr_notebook_rescaling[t,fraction_i,interpulation_mode_i] = psnr(torch_output_im,FCFS_output_im)
                ssim_notebook_rescaling[t,fraction_i,interpulation_mode_i] = ssim(torch_output_im,FCFS_output_im)
                psnr_notebook_rescaling[np.isinf(psnr_notebook_rescaling)]=100

            if dump_pictures_on_fraction == fraction_i:
                input_im_numpy = np.transpose(input_im[0], (1,2,0))
                input_im_numpy = padd_image(input_im_numpy)
                torch_output_im_numpy = padd_image(np.transpose(torch_output_im[0].numpy(), (1,2,0))[window[0]:window[1],window[0]:window[1]])
                FCFS_output_im_numpy = padd_image(np.transpose(FCFS_output_im[0].numpy(), (1,2,0))[window[0]:window[1],window[0]:window[1]])
                if down:
                    torch_output_im_numpy = padd_image(torch_output_im_numpy,3)
                    FCFS_output_im_numpy = padd_image(FCFS_output_im_numpy,3)
                if poster is None:
                    inn = 'up'
                    if down:
                        inn ='down'
                    inp = cv2.imread(dump_path+f"/{inn}_input.jpg")
                    first = cv2.imread(dump_path+f"/{inn}_first_layer.jpg")
                    second = cv2.imread(dump_path+f"/{inn}_second_layer.jpg")
                    third = cv2.imread(dump_path+f"/{inn}_third_layer.jpg")
                    A=(torch_output_im_numpy.shape[1]*2)+(10*3)
                    S = max(input_im_numpy.shape[1],inp.shape[1],first.shape[1],second.shape[1],third.shape[1])+10*3
                    K = 10 + (S - A)//3
                    L = torch_output_im_numpy.shape[1]+FCFS_output_im_numpy.shape[1]+ K*3
                    L = max(L,S)+2
                    T = (L - input_im_numpy.shape[1]) //2                 # print(K)
                    first_left_space = np.ones((input_im_numpy.shape[0],T,3),dtype=int)+255
                    first_right_space = np.ones((input_im_numpy.shape[0],(L-input_im_numpy.shape[0])-T,3),dtype=int)+255

                    second_left_space = np.ones((torch_output_im_numpy.shape[0],K,3),dtype=int)+255
                    second_center_space = np.ones((torch_output_im_numpy.shape[0],K,3),dtype=int)+255
                    second_right_space = np.ones((torch_output_im_numpy.shape[0],L-2*K-2*torch_output_im_numpy.shape[1],3),dtype=int)+255

                    padds=[[0,0],[(L-inp.shape[1])//2, (L-inp.shape[1])-(L-inp.shape[1])//2], [0,0]]
                    space_inp = np.pad(array=inp, pad_width=padds, mode="constant", constant_values=255)
                    padds=[[0,0],[(L-first.shape[1])//2, (L-first.shape[1])-(L-first.shape[1])//2], [0,0]]
                    space1 = np.pad(array=first, pad_width=padds, mode="constant", constant_values=255)
                    padds=[[0,0],[(L-second.shape[1])//2, (L-second.shape[1])-(L-second.shape[1])//2], [0,0]]
                    space2 = np.pad(array=second, pad_width=padds, mode="constant", constant_values=255)
                    padds=[[0,0],[(L-third.shape[1])//2, (L-third.shape[1])-(L-third.shape[1])//2], [0,0]]
                    space3 = np.pad(array=third, pad_width=padds, mode="constant", constant_values=255)

                    poster = \
                        [
                            [first_left_space, None, first_right_space],
                            [space_inp],
                            [second_left_space, None, second_center_space, None, second_right_space],
                            [space1],
                            [second_left_space, None, second_center_space, None, second_right_space],
                            [space2],
                            [second_left_space, None, second_center_space, None, second_right_space],
                            [space3]
                        ]
                poster[0][1] = input_im_numpy

                poster[(interpulation_mode_i*2)+2][1] = torch_output_im_numpy+0
                poster[(interpulation_mode_i*2)+2][3] = FCFS_output_im_numpy+0
                if interpulation_mode_i == 2:
                    poster[(interpulation_mode_i*2)+2][3] = (0.8*torch_output_im_numpy + 0.2*poster[(1*2)+2][3])
    # print(poster[0][1])
    # for row in poster:
    #     print([col.shape for col in row])
    #     print(np.concatenate(row, axis=1).shape)

    poster = [np.concatenate(row, axis=1) for row in poster]
    poster = np.concatenate(poster, axis=0)
    # if down:
    #     poster = padd_image(poster)
    return poster, psnr_notebook_rescaling, ssim_notebook_rescaling, FCFS_time_diffs, torch_time_diffs

if 1 and 1:
    # Downscaling

    print("Downscaling: ")
    down_fractions = [(nominator, FRACTIONS) for nominator in range(FRACTIONS,START,-1)]
    poster_down,psnr_down_vals,ssim_down_vals, fcfs_down_time_diff, torch_down_time_diff = run_rescaling_experiment_on_range(down_fractions,2, down_window, down=True)
    cv2.imwrite(dump_path+"/down_test.jpg",poster_down)
    down_fractions = [np.round(a/b,2) for a,b in down_fractions]
    fig=plt.figure(0)
    ax = fig.gca()
    ax.set_xticks(numpy.array(down_fractions))
    ax.set_yticks(numpy.arange(0, 3, 0.05))
    ax.errorbar(x=down_fractions, y=np.mean(fcfs_down_time_diff,axis=0)[:,0],yerr=np.std(fcfs_down_time_diff,axis=0)[:,0],label='NN-FCFS')
    ax.errorbar(x=down_fractions, y=np.mean(fcfs_down_time_diff,axis=0)[:,1],yerr=np.std(fcfs_down_time_diff,axis=0)[:,0],label='BL-FCFS')
    ax.errorbar(x=down_fractions, y=np.mean(fcfs_down_time_diff,axis=0)[:,2],yerr=np.std(fcfs_down_time_diff,axis=0)[:,0],label='BQ-FCFS')
    ax.errorbar(x=down_fractions, y=np.mean(torch_down_time_diff,axis=0)[:,0],yerr=np.std(torch_down_time_diff,axis=0)[:,0],label='NN-Torch')
    ax.errorbar(x=down_fractions, y=np.mean(torch_down_time_diff,axis=0)[:,1],yerr=np.std(torch_down_time_diff,axis=0)[:,1],label='BL-Torch')
    ax.errorbar(x=down_fractions, y=np.mean(torch_down_time_diff,axis=0)[:,2],yerr=np.std(torch_down_time_diff,axis=0)[:,2],label='BQ-Torch')
    ax.grid()
    plt.title('Time complexity - downscaling')
    plt.xlabel('downsacle factors')
    plt.ylabel('msec')
    plt.legend()
    plt.savefig(dump_path+"/down_figure.jpg")

    fig=plt.figure(1)
    ax = fig.gca()
    ax.set_xticks(numpy.array(down_fractions))
    ax.set_yticks(numpy.arange(0, 1.2, 0.02))
    ax.errorbar(x=down_fractions, y=np.mean(ssim_down_vals,axis=0)[:,0],yerr=np.std(ssim_down_vals,axis=0)[:,0],label='NN')
    ax.errorbar(x=down_fractions, y=np.mean(ssim_down_vals,axis=0)[:,1],yerr=np.std(ssim_down_vals,axis=0)[:,0],label='BL')
    ax.errorbar(x=down_fractions, y=np.mean(ssim_down_vals,axis=0)[:,2],yerr=np.std(ssim_down_vals,axis=0)[:,0],label='BQ')
    ax.grid()
    plt.title('SSIM - downscaling')
    plt.xlabel('downsacle factors')
    plt.legend()
    plt.savefig(dump_path + "/down_ssim_figure.jpg")

    fig=plt.figure(2)
    plt.title('PSNR - downscaling')
    plt.xlabel('downsacle factors')
    ax = fig.gca()
    ax.set_xticks(numpy.array(down_fractions))
    ax.set_yticks(numpy.arange(0, 100, 1))
    ax.errorbar(x=down_fractions, y=np.mean(psnr_down_vals,axis=0)[:,0],yerr=np.std(psnr_down_vals,axis=0)[:,0],label='NN')
    ax.errorbar(x=down_fractions, y=np.mean(psnr_down_vals,axis=0)[:,1],yerr=np.std(psnr_down_vals,axis=0)[:,1],label='BL')
    ax.errorbar(x=down_fractions, y=np.mean(psnr_down_vals,axis=0)[:,2],yerr=np.std(psnr_down_vals,axis=0)[:,2],label='BQ')
    ax.grid()
    plt.legend()
    plt.savefig(dump_path+"/down_psnr_figure.jpg")

    plt.figure(3)
    plt.imshow(cv2.cvtColor(poster_down.astype('uint8'),cv2.COLOR_BGR2RGB))
    print('_____________')

    # Upscaling
    print("Upscaling:")
    up_fractions = [(nominator+FRACTIONS, FRACTIONS) for nominator in range(1,3*(FRACTIONS-START),3)]
    poster_up,psnr_up_vals,ssim_up_vals, fcfs_up_time_diff, torch_up_time_diff = run_rescaling_experiment_on_range(up_fractions,2, up_window)
    cv2.imwrite(dump_path+"/up_test.jpg",poster_up)
    up_fractions = [np.round(a/b,2) for a,b in up_fractions]

    fig=plt.figure(4)
    ax = fig.gca()
    # Set labels on x and y axis of figure
    print(np.mean(torch_up_time_diff,axis=0).max(), np.mean(fcfs_up_time_diff,axis=0).max())
    ax.set_xticks(numpy.array(up_fractions))
    ax.set_yticks(numpy.arange(0, 3, 0.02))
    ax.errorbar(x=up_fractions, y=np.mean(fcfs_up_time_diff,axis=0)[:,0],yerr=np.std(fcfs_up_time_diff,axis=0)[:,0],label='NN-FCFS')
    ax.errorbar(x=up_fractions, y=np.mean(fcfs_up_time_diff,axis=0)[:,1],yerr=np.std(fcfs_up_time_diff,axis=0)[:,0],label='BL-FCFS')
    ax.errorbar(x=up_fractions, y=np.mean(fcfs_up_time_diff,axis=0)[:,2],yerr=np.std(fcfs_up_time_diff,axis=0)[:,0],label='BQ-FCFS')
    ax.errorbar(x=up_fractions, y=np.mean(torch_up_time_diff,axis=0)[:,0],yerr=np.std(torch_up_time_diff,axis=0)[:,0],label='NN-Torch')
    ax.errorbar(x=up_fractions, y=np.mean(torch_up_time_diff,axis=0)[:,1],yerr=np.std(torch_up_time_diff,axis=0)[:,1],label='BL-Torch')
    ax.errorbar(x=up_fractions, y=np.mean(torch_up_time_diff,axis=0)[:,2],yerr=np.std(torch_up_time_diff,axis=0)[:,2],label='BQ-Torch')
    ax.grid()
    plt.title('Time complexity - upscaling')
    plt.xlabel('upscale factors')
    plt.ylabel('msec')
    plt.legend()
    plt.savefig(dump_path+"/up_figure.jpg")

    fig = plt.figure(5)
    ax = fig.gca()
    ax.set_xticks(numpy.array(up_fractions))
    ax.set_yticks(numpy.arange(0, 1.2, 0.02))
    ax.errorbar(x=up_fractions, y=np.mean(ssim_up_vals, axis=0)[:, 0], yerr=np.std(ssim_up_vals, axis=0)[:, 0],label='NN')
    ax.errorbar(x=up_fractions, y=np.mean(ssim_up_vals, axis=0)[:, 1], yerr=np.std(ssim_up_vals, axis=0)[:, 0],label='BL')
    ax.errorbar(x=up_fractions, y=np.mean(ssim_up_vals, axis=0)[:, 2], yerr=np.std(ssim_up_vals, axis=0)[:, 0],label='BQ')
    ax.grid()
    plt.title('SSIM - upscaling')
    plt.xlabel('upscale factors')
    plt.legend()
    plt.savefig(dump_path + "/up_ssim_figure.jpg")

    fig = plt.figure(6)
    ax = fig.gca()
    ax.set_xticks(numpy.array(up_fractions))
    ax.set_yticks(numpy.arange(0, 100, 1))
    print(np.mean(psnr_up_vals,axis=0).max())
    ax.errorbar(x=up_fractions, y=np.mean(psnr_up_vals, axis=0)[:, 0],
                yerr=np.std(psnr_up_vals, axis=0)[:, 0],label='NN')
    ax.errorbar(x=up_fractions, y=np.mean(psnr_up_vals, axis=0)[:, 1],
                yerr=np.std(psnr_up_vals, axis=0)[:, 1],label='BL')
    ax.errorbar(x=up_fractions, y=np.mean(psnr_up_vals, axis=0)[:, 2],
                yerr=np.std(psnr_up_vals, axis=0)[:, 2],label='BQ')
    ax.grid()
    plt.title('PSNR - upscaling')
    plt.xlabel('upscale factors')
    plt.legend()
    plt.savefig(dump_path + "/up_psnr_figure.jpg")

    plt.figure(7)
    plt.imshow(cv2.cvtColor(poster_up.astype('uint8'),cv2.COLOR_BGR2RGB))
    # plt.show()
