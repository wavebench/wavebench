import os

# Gaussian lens

## U-Net
os.system('python train_unet_is.py --medium_type gaussian_lens --channel_reduction_factor 2 --gpu_devices 0')
os.system('python train_unet_is.py --medium_type gaussian_lens --channel_reduction_factor 1 --gpu_devices 0')

# FNO
os.system('python train_fno_is.py --medium_type gaussian_lens --num_layers 4 --gpu_devices 0')
os.system('python train_fno_is.py --medium_type gaussian_lens --num_layers 8 --gpu_devices 0')


# Gaussian Random Field

## U-Net
os.system('python train_unet_is.py --medium_type gaussian_random_field --channel_reduction_factor 2 --gpu_devices 0')
os.system('python train_unet_is.py --medium_type gaussian_random_field --channel_reduction_factor 1 --gpu_devices 0')

## FNO
os.system('python train_fno_is.py --medium_type gaussian_random_field --num_layers 4 --gpu_devices 0')
os.system('python train_fno_is.py --medium_type gaussian_random_field --num_layers 8 --gpu_devices 0')
