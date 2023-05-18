import os

# Gaussian lens

gpu_devices = 1
for kernel_type in ['isotropic', 'anisotropic']:
  for frequency in [1.0, 1.5, 2.0, 4.0]:
    # FNO
    # os.system(f'python train_fno_helmholtz.py --kernel_type {kernel_type} --num_layers 8 --gpu_devices {gpu_devices}')

    ## U-Net
    os.system(f'python train_unet_helmholtz.py --kernel_type {kernel_type}  --channel_reduction_factor 1 --gpu_devices {gpu_devices}')

