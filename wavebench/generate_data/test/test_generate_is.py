# %%
import os
import ml_collections
import numpy as np
import cv2
import matlab.engine # the matlab engine for python
import matplotlib.pyplot as plt

import jax

from wavebench import wavebench_path
from wavebench.generate_data.time_varying.generate_data_is import generate_is
from wavebench import wavebench_dataset_path
from wavebench.generate_data.time_varying.gaussian_random_field import generate_gaussian_random_field
from wavebench.utils import absolute_file_paths, seed_everything


# %%
thick_lines_data_path = os.path.join(
    wavebench_dataset_path, "time_varying/thick_lines")

config = ml_collections
config.device_id = 0
config.save_data = False

config.domain_sidelen = 512
config.domain_dx = 2

config.medium_type = 'gaussian_lens'
# config.medium_type = 'gaussian_random_field'

config.medium_source_loc = (199, 219)
config.medium_density = 2650
config.pml_size = 10

#  define the properties of the propagation medium
min_wavespeed = 1400
max_wavespeed = 4000
point_mass_strength = -31000

if config.medium_type == 'gaussian_lens':
  z = np.ones((config.domain_sidelen,config.domain_sidelen))
  z[config.medium_source_loc] = point_mass_strength
  medium_sound_speed = cv2.GaussianBlur(
      z,
      ksize=(0, 0),
      sigmaX=200,
      borderType=cv2.BORDER_REPLICATE)
elif config.medium_type == 'gaussian_random_field':
  seed_everything(42)
  medium_sound_speed = generate_gaussian_random_field(
      size = config.domain_sidelen,
      alpha=3.0)
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
kwave_measurements = np.array(kwave_measurements).T

# %%
mse = np.mean( (kwave_measurements - jwave_measurements)**2 )
np.testing.assert_array_less(mse, 1e-4)

# %%
fig, axes = plt.subplots(1, 3, figsize=(8, 6))

axes[0].imshow(kwave_measurements)
axes[0].set_title('kwave')
axes[1].imshow(jwave_measurements)
axes[1].set_title('jwave')
axes[2].imshow(np.abs(kwave_measurements - jwave_measurements))
axes[2].set_title('diff')



