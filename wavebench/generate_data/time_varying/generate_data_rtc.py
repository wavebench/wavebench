""" Generate data for Reverse Time Continuation (RTC) dataset."""
import os
import cv2
import numpy as np
import ml_collections
from tqdm import tqdm
from functools import partial

import jax
from jax import jit
from jax import numpy as jnp

from jwave import FourierSeries
from jwave.utils import load_image_to_numpy
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import Medium, Domain, TimeAxis

from wavebench.generate_data.time_varying.gaussian_random_field import generate_gaussian_random_field
from wavebench import wavebench_dataset_path
from wavebench.utils import absolute_file_paths, seed_everything


def generate_rtc(config):
  jax.config.update(
    "jax_default_device", jax.devices()[config.device_id])

  print('Save data?', config.save_data)
  domain = Domain(
    (config.domain_sidelen, config.domain_sidelen),
    (config.domain_dx, config.domain_dx))

  medium = Medium(
    domain=domain,
    sound_speed=config.medium_sound_speed[..., np.newaxis])
  medium.density = config.medium_density
  medium.pml_size = config.pml_size

  time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=0.2)


  @jit
  def compute_final_pressure(medium, initial_pressure):
    final_pressure = simulate_wave_propagation(
        medium, time_axis, p0=initial_pressure)
    return final_pressure.on_grid.squeeze()[-1]


  num_data = len(config.source_list)
  rtc_dataset = os.path.join(wavebench_dataset_path, "time_varying/rtc")

  if config.save_data:
    # https://vmascagn.web.cern.ch/vmascagn/LABO_2020/numpy-memmap_for_ghost_imaging.html
    initial_pressure_dataset = np.memmap(
        f'{rtc_dataset}/{config.medium_type}_initial_pressure_dataset.npy',
        mode='w+',
        shape=(num_data, config.domain_sidelen, config.domain_sidelen),
            dtype=np.float32)
    final_pressure_dataset = np.memmap(
        f'{rtc_dataset}/{config.medium_type}_final_pressure_dataset.npy',
        mode='w+',
        shape=(num_data, config.domain_sidelen, config.domain_sidelen),
            dtype=np.float32)
  else:
    initial_pressure_dataset = np.zeros(
      (num_data, config.domain_sidelen, config.domain_sidelen))
    final_pressure_dataset = np.zeros(
      (num_data, config.domain_sidelen, config.domain_sidelen))

  for (idx, image) in enumerate(tqdm(config.source_list)):
    initial_pressure = load_image_to_numpy(image,
        image_size=(config.domain_sidelen, config.domain_sidelen))/255

    initial_pressure_dataset[idx, ...] = initial_pressure
    if config.save_data:
      initial_pressure_dataset.flush()

    initial_pressure = jnp.expand_dims(initial_pressure, -1)
    initial_pressure = FourierSeries(initial_pressure, domain)
    final_pressure = compute_final_pressure(medium, initial_pressure)
    final_pressure_dataset[idx, ...] = final_pressure
    if config.save_data:
      final_pressure_dataset.flush()

  return initial_pressure_dataset, final_pressure_dataset

if __name__ == '__main__':

  thick_lines_data_path = os.path.join(
      wavebench_dataset_path, "time_varying/thick_lines")

  config = ml_collections

  config.domain_sidelen = 512
  config.domain_dx = 2

  # config.medium_type = 'gaussian_lens'
  config.medium_type = 'gaussian_random_field'

  config.medium_source_loc = (199, 219)
  config.medium_density = 2650
  config.pml_size = 10

  #  define properties of the propagation medium
  min_wavespeed = 1400
  max_wavespeed = 4000

  if config.medium_type == 'gaussian_lens':
    point_mass_strength = -31000
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

  config.source_list = sorted(absolute_file_paths(thick_lines_data_path))#[:10]
  config.device_id = 0 #1
  config.save_data = True

  generate_rtc(config)
