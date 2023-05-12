""" Generate data for Reverse Time Continuation (RTC) dataset.

Example usage:
python generate_data_rtc.py \
  --device_id 0 \
  --medium_type gaussian_lens

python generate_data_rtc.py \
  --device_id 0 \
  --medium_type gaussian_random_field
"""

import os
import argparse

import cv2
import numpy as np
import ml_collections
from tqdm import tqdm

import jax
from jax import jit
from jax import numpy as jnp

from jwave import FourierSeries
from jwave.utils import load_image_to_numpy
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import Medium, Domain, TimeAxis

from wavebench import wavebench_dataset_path
from wavebench.utils import absolute_file_paths

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
        medium, time_axis, p0=initial_pressure,
        )
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


thick_lines_data_path = os.path.join(
    wavebench_dataset_path, "time_varying/thick_lines")


parser = argparse.ArgumentParser()
parser.add_argument('--medium_type', type=str, default='gaussian_lens',
                    help='Can be `gaussian_lens` or `gaussian_random_field`.')
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--save_data', default=True,
                    type=lambda x: (str(x).lower() == 'true'))


def main():

  args = parser.parse_args()
  config = ml_collections

  config.medium_type = args.medium_type
  config.device_id = args.device_id
  config.save_data = args.save_data

  config.domain_sidelen = 128
  config.domain_dx = 8
  # the above seetings give a domain of 1024 km x 1024 km

  config.medium_source_loc = (50, 55)
  config.medium_density = 2650
  config.pml_size = 2

  min_wavespeed = 1400 # [m/s]
  max_wavespeed = 4000 # [m/s]

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

  config.source_list = sorted(absolute_file_paths(thick_lines_data_path))#[:10]
  print(f'Number of sources: {len(config.source_list)}')
  generate_rtc(config)

if __name__ == '__main__':
  main()
