import os

# Gaussian lens

## UNO
os.system('python train_uno_rtc.py --medium_type gaussian_lens --modes 12 --gpu_devices 1')
os.system('python train_uno_rtc.py --medium_type gaussian_lens --modes 16 --gpu_devices 1')


# # FNO
# os.system('python train_fno_rtc.py --medium_type gaussian_lens --num_layers 8 --gpu_devices 1')
# os.system('python train_fno_rtc.py --medium_type gaussian_lens --num_layers 4 --gpu_devices 1')

# ## U-Net
# os.system('python train_unet_rtc.py --medium_type gaussian_lens  --channel_reduction_factor 1 --gpu_devices 1')
# os.system('python train_unet_rtc.py --medium_type gaussian_lens  --channel_reduction_factor 2 --gpu_devices 1')


# Gaussian Random Field (isotropic)

# UNO
os.system('python train_uno_rtc.py --medium_type grf_isotropic --modes 12 --gpu_devices 1')
os.system('python train_uno_rtc.py --medium_type grf_isotropic --modes 16 --gpu_devices 1')


# ## FNO
# os.system('python train_fno_rtc.py --medium_type grf_isotropic  --num_layers 8 --gpu_devices 1')
# os.system('python train_fno_rtc.py --medium_type grf_isotropic  --num_layers 4 --gpu_devices 1')


# ## U-Net
# os.system('python train_unet_rtc.py --medium_type grf_isotropic --channel_reduction_factor 1 --gpu_devices 1')
# os.system('python train_unet_rtc.py --medium_type grf_isotropic --channel_reduction_factor 2 --gpu_devices 1')

# Gaussian Random Field (anisotropic)

# UNO
os.system('python train_uno_rtc.py --medium_type grf_anisotropic --modes 12 --gpu_devices 1')
os.system('python train_uno_rtc.py --medium_type grf_anisotropic --modes 16 --gpu_devices 1')


# ## FNO
# os.system('python train_fno_rtc.py --medium_type grf_anisotropic  --num_layers 8 --gpu_devices 1')
# os.system('python train_fno_rtc.py --medium_type grf_anisotropic  --num_layers 4 --gpu_devices 1')


# ## U-Net
# os.system('python train_unet_rtc.py --medium_type grf_anisotropic --channel_reduction_factor 1 --gpu_devices 1')
# os.system('python train_unet_rtc.py --medium_type grf_anisotropic --channel_reduction_factor 2 --gpu_devices 1')
