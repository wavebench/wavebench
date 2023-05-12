import os

# Gaussian lens

## U-Net
os.system('python train_unet_rtc.py --medium_type gaussian_lens  --gpu_devices 1')

# FNO
os.system('python train_fno_rtc.py --medium_type gaussian_lens --gpu_devices 1')


# Gaussian Random Field

## U-Net
os.system('python train_unet_rtc.py --medium_type gaussian_random_field --gpu_devices 1')

## FNO
os.system('python train_fno_rtc.py --medium_type gaussian_random_field  --gpu_devices 1')