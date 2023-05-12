# %%
import os
import ml_collections
import numpy as np
import cv2
import matlab.engine # the matlab engine for python
import jax

from wavebench import wavebench_path
from wavebench.generate_data.time_varying.generate_data_is import generate_is
from wavebench import wavebench_dataset_path
from wavebench.utils import absolute_file_paths
from wavebench.plot_utils import plot_images, remove_frame


# %%
thick_lines_data_path = os.path.join(
    wavebench_dataset_path, "time_varying/thick_lines")

config = ml_collections
config.save_data = False
# config.medium_type = 'gaussian_lens'
config.medium_type = 'gaussian_random_field'
config.device_id = 0

config.domain_sidelen = 128
config.domain_dx = 8
# the above seeting gives a domain of 1024 km x 1024 km

config.medium_source_loc = (50, 55)
config.medium_density = 2650
config.pml_size = 2

#  define the properties of the propagation medium
min_wavespeed = 1400 # [m/s]
max_wavespeed = 4000 # [m/s]
point_mass_strength = -31000

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
config.source_list = sorted(absolute_file_paths(thick_lines_data_path))[:1]
initial_pressure_dataset, boundary_measurement_dataset = generate_is(config)
jwave_measurements = boundary_measurement_dataset[0]


resized_len = config.domain_sidelen//2

resized_initial_pressure = np.zeros_like(initial_pressure_dataset[0])

image_array = jax.image.resize(
  initial_pressure_dataset[0],
  (resized_len, resized_len),
  method='bicubic')

resized_initial_pressure[
    :resized_len,
    resized_len//2: resized_len//2 + resized_len] = image_array


eng = matlab.engine.start_matlab()
eng.cd(str(os.path.join(wavebench_path, "wavebench/generate_data/test")))


kwave_measurements = eng.compute_is_measurements(
  np.double(config.medium_sound_speed),
  np.double(config.medium_density),
  np.double(config.domain_dx),
  resized_initial_pressure,
  np.double(config.pml_size))
kwave_measurements = np.array(kwave_measurements).T[1:, :]
# To make k-wave and j-wave comparable, the k-Wave simulation was ran for
# one extra time step; we remove the extra time step here.


# %%
mse = np.mean( (kwave_measurements - jwave_measurements)**2 )

fig, axes = plot_images(
  [kwave_measurements,
   jwave_measurements,
   np.abs(kwave_measurements - jwave_measurements)],
  cbar='one',
  fig_size=(6, 3),
  cmap='coolwarm')

axes[0].set_title('kwave')
axes[1].set_title('jwave')
axes[2].set_title(f'diff mse={mse:.2}')

[remove_frame(ax) for ax in axes.flatten()];

