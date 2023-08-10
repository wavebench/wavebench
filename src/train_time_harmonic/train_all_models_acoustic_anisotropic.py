import os

gpu_devices = 0
kernel_type = 'anisotropic'
num_epochs = 20
for frequency in [10, 15, 20, 40]:
  # FNO
  os.system(f'python train_fno_helmholtz.py --frequency {frequency} --kernel_type {kernel_type} --num_layers 4 --gpu_devices {gpu_devices} --num_epochs {num_epochs}')
  os.system(f'python train_fno_helmholtz.py --frequency {frequency} --kernel_type {kernel_type} --num_layers 8 --gpu_devices {gpu_devices} --num_epochs {num_epochs}')

  # U-Net
  os.system(f'python train_unet_helmholtz.py --frequency {frequency} --kernel_type {kernel_type}  --channel_reduction_factor 2 --gpu_devices {gpu_devices} --num_epochs {num_epochs}')
  os.system(f'python train_unet_helmholtz.py --frequency {frequency} --kernel_type {kernel_type}  --channel_reduction_factor 1 --gpu_devices {gpu_devices} --num_epochs {num_epochs}')

  # UNO
  os.system(f'python train_uno_helmholtz.py --frequency {frequency} --kernel_type {kernel_type} --modes 12 --gpu_devices {gpu_devices} --num_epochs {num_epochs}')
  os.system(f'python train_uno_helmholtz.py --frequency {frequency} --kernel_type {kernel_type} --modes 16 --gpu_devices {gpu_devices} --num_epochs {num_epochs}')
