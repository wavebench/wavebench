import os

# Gaussian lens

## U-Net
os.system('python train_unet_is.py --medium_type gaussian_lens --gpu_devices 0')

# FNO
os.system('python train_fno_is.py --medium_type gaussian_lens --gpu_devices 0')


# Gaussian Random Field

## U-Net
os.system('python train_unet_is.py --medium_type gaussian_random_field --gpu_devices 0')

## FNO
os.system('python train_fno_is.py --medium_type gaussian_random_field --gpu_devices 0')