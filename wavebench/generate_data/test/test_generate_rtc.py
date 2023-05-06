# %%
import os
import ml_collections
import numpy as np
import cv2
import matlab.engine # the matlab engine for python
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt

from wavebench.generate_data.time_varying.generate_data_rtc import generate_rtc
from wavebench import wavebench_dataset_path
from wavebench.utils import absolute_file_paths
from wavebench import utils

# %%
thick_lines_data_path = os.path.join(
    wavebench_dataset_path, "time_varying/thick_lines")

config = ml_collections
config.save_data = False

config.domain_sidelen = 512
config.domain_dx = 2

config.use_gaussian_lens_medium = True
config.medium_source_loc = (199, 219)
config.medium_density = 2650
config.pml_size = 10

#  define the properties of the propagation medium
min_wavespeed = 1400
max_wavespeed = 4000
point_mass_strength = -31000

if config.use_gaussian_lens_medium:
    z = np.ones((config.domain_sidelen,config.domain_sidelen))
    z[config.medium_source_loc] = point_mass_strength
    medium_sound_speed = cv2.GaussianBlur(
        z,
        ksize=(0, 0),
        sigmaX=200,
        borderType=cv2.BORDER_REPLICATE)
else:
    raise NotImplementedError

medium_sound_speed -= medium_sound_speed.min()
medium_sound_speed /= medium_sound_speed.max()

config.medium_sound_speed = medium_sound_speed*(
max_wavespeed - min_wavespeed) + min_wavespeed

# only a single example is generated
config.source_list = sorted(absolute_file_paths(thick_lines_data_path))[:1]
initial_pressure_dataset, final_pressure_dataset = generate_rtc(config)
jwave_final_pressure = final_pressure_dataset[0]

# %%

eng = matlab.engine.start_matlab()

kwave_final_pressure = eng.compute_rtc_final(
  np.double(config.medium_sound_speed),
  np.double(config.medium_density),
  np.double(config.domain_dx),
  initial_pressure_dataset[0])
kwave_final_pressure = np.array(kwave_final_pressure)


# %%
assert_allclose(
  kwave_final_pressure,
  jwave_final_pressure, atol=0.1)

# %%
fig, axes = plt.subplots(1, 3, figsize=(9, 3))

axes[0].imshow(kwave_final_pressure)
axes[0].set_title('kwave')
axes[1].imshow(jwave_final_pressure)
axes[1].set_title('jwave')
axes[2].imshow(np.abs(jwave_final_pressure - kwave_final_pressure))
axes[2].set_title('diff')



