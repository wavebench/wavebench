# %%
import os
import ml_collections
import numpy as np
import cv2
import matlab.engine # the matlab engine for python
import jax

import matplotlib.pyplot as plt
from wavebench.generate_data.time_varying.generate_data_rtc import generate_rtc
from wavebench import wavebench_dataset_path
from wavebench.utils import absolute_file_paths
from wavebench import wavebench_path
from wavebench.plot_utils import plot_images, remove_frame

# %%

config = ml_collections
# config.initial_pressure_type = 'thick_lines'
config.initial_pressure_type = 'mnist'

config.save_data = False
# config.medium_type = 'gaussian_random_field'
config.medium_type = 'gaussian_lens'
config.device_id = 0


config.domain_sidelen = 128
config.domain_dx = 8
# the above seetings give a domain of 1024 km x 1024 km

config.medium_source_loc = (50, 55)
config.medium_density = 2650
config.pml_size = 2

#  define the properties of the propagation medium
min_wavespeed = 1400 # [m/s]
max_wavespeed = 4000 # [m/s]
point_mass_strength = -31000

# thick_lines_data_path = os.path.join(
#     wavebench_dataset_path, "time_varying/thick_lines")

data_path = os.path.join(
    wavebench_dataset_path,
    f"time_varying/{config.initial_pressure_type}")


if config.medium_type == 'gaussian_lens':
  z = np.ones((config.domain_sidelen,config.domain_sidelen))
  z[config.medium_source_loc] = point_mass_strength
  medium_sound_speed = cv2.GaussianBlur(
      z,
      ksize=(0, 0),
      sigmaX=50,
      borderType=cv2.BORDER_REPLICATE)
elif config.medium_type == 'gaussian_random_field':
  medium_sound_speed = np.fromfile(
    os.path.join(
      wavebench_dataset_path, "time_varying/wavespeed/cp_128x128_00001.H@"),
    dtype=np.float32).reshape(128, 128)

  if config.domain_sidelen != 128:
    medium_sound_speed = jax.image.resize(
        medium_sound_speed,
        (config.domain_sidelen, config.domain_sidelen),
        'bicubic')
else:
  raise NotImplementedError

medium_sound_speed -= medium_sound_speed.min()
medium_sound_speed /= medium_sound_speed.max()

config.medium_sound_speed = medium_sound_speed*(
max_wavespeed - min_wavespeed) + min_wavespeed

# only a single example is generated
config.source_list = sorted(absolute_file_paths(data_path))[:1]#[82:83]
initial_pressure_dataset, final_pressure_dataset = generate_rtc(config)
jwave_final_pressure = final_pressure_dataset[0]

# %%

eng = matlab.engine.start_matlab()
eng.cd(str(os.path.join(wavebench_path, "wavebench/generate_data/test")))

# MATLAB expects the input arguments to be of data type double
kwave_final_pressure = eng.compute_rtc_final(
  np.double(config.medium_sound_speed),
  np.double(config.medium_density),
  np.double(config.domain_dx),
  initial_pressure_dataset[0],
  np.double(config.pml_size),
)
kwave_final_pressure = np.array(kwave_final_pressure)


# %%
mse = np.mean( (kwave_final_pressure - jwave_final_pressure)**2 )
np.testing.assert_array_less(mse, 1e-4)
print(mse)

# %%

fig, axes = plot_images(
  [kwave_final_pressure,
   jwave_final_pressure,
   np.abs(jwave_final_pressure - kwave_final_pressure)],
  cbar='one',
  # vrange='individual',
  fig_size=(9, 3),
  cmap='coolwarm')

axes[0].set_title('kwave')
axes[1].set_title('jwave')
axes[2].set_title(f'diff mse={mse:.2}')

[remove_frame(ax) for ax in axes.flatten()]

# %%

plt.imshow(config.medium_sound_speed, cmap='coolwarm')

# %%
