""" Generate data for Inverse Source (IS) dataset.
Example usage:
python generate_data_is.py \
  --device_id 1 \
  --medium_type gaussian_lens

python generate_data_is.py \
  --device_id 1 \
  --medium_type gaussian_random_field

python generate_data_is.py \
  --device_id 0 \
  --initial_pressure_type mnist \
  --medium_type gaussian_lens

python generate_data_is.py \
  --device_id 0 \
  --initial_pressure_type mnist \
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
from wavebench.utils import absolute_file_paths#, seed_everything


def generate_is(config):
  jax.config.update(
    "jax_default_device", jax.devices()[config.device_id])
  print(config.device_id)
  print(jax.devices())

  print('Save data?', config.save_data)
  domain = Domain(
    (config.domain_sidelen, config.domain_sidelen),
    (config.domain_dx, config.domain_dx))

  medium = Medium(
    domain=domain,
    sound_speed=config.medium_sound_speed[..., np.newaxis])
  medium.density = config.medium_density
  medium.pml_size = config.pml_size

  resized_len = config.domain_sidelen//2
  time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=0.2)


  def sensor_func(p, u, rho):
    return p.on_grid[medium.pml_size, ...].squeeze()

  @jit
  def compute_measurements_from_sensors(medium, initial_pressure):
    records = simulate_wave_propagation(
        medium, time_axis, p0=initial_pressure, sensors=sensor_func)
    return records


  num_data = len(config.source_list)
  is_dataset = os.path.join(wavebench_dataset_path, "time_varying/is")

  if config.save_data:
    # https://vmascagn.web.cern.ch/vmascagn/LABO_2020/numpy-memmap_for_ghost_imaging.html
    initial_pressure_dataset = np.memmap(
        f'{is_dataset}/{config.initial_pressure_type}_{config.medium_type}_initial_pressure_dataset.npy',  # pylint: disable=line-too-long
        mode='w+',
        shape=(num_data, config.domain_sidelen, config.domain_sidelen),
            dtype=np.float32)
    boundary_measurement_dataset = np.memmap(
        f'{is_dataset}/{config.initial_pressure_type}_{config.medium_type}_boundary_measurement_dataset.npy',  # pylint: disable=line-too-long
        mode='w+',
        shape=(num_data, int(time_axis.Nt), config.domain_sidelen),
            dtype=np.float32)
  else:
    initial_pressure_dataset = np.zeros(
      (num_data, config.domain_sidelen, config.domain_sidelen))
    boundary_measurement_dataset = np.zeros(
      (num_data, int(time_axis.Nt), config.domain_sidelen))

  for (idx, image) in enumerate(tqdm(config.source_list)):
    image_array = load_image_to_numpy(image,
        image_size=(config.domain_sidelen, config.domain_sidelen))/255

    initial_pressure_dataset[idx, ...] = image_array
    if config.save_data:
      initial_pressure_dataset.flush()

    # Put the image array at the top center of the domain, so that the objects
    # are closer to the sensor; this make the problem less ill-posed.
    initial_pressure = np.zeros_like(image_array)
    image_array = jax.image.resize(
      image_array,
      (resized_len, resized_len),
      method='bicubic')

    initial_pressure[
        :resized_len,
        resized_len//2: resized_len//2 + resized_len] = image_array
    initial_pressure = jnp.expand_dims(
        initial_pressure, -1)
    initial_pressure = FourierSeries(
        initial_pressure, domain)

    measurements = compute_measurements_from_sensors(
      medium, initial_pressure)

    boundary_measurement_dataset[idx, ...] = measurements
    if config.save_data:
      boundary_measurement_dataset.flush()

  return initial_pressure_dataset, boundary_measurement_dataset



parser = argparse.ArgumentParser()
parser.add_argument('--medium_type', type=str, default='gaussian_lens',
                    help='Can be `gaussian_lens` or `gaussian_random_field`.')
parser.add_argument('--initial_pressure_type', type=str, default='thick_lines',
                    help='Can be `thick_lines` or `mnist`.')
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--save_data', default=True,
                    type=lambda x: (str(x).lower() == 'true'))

def main():

  args = parser.parse_args()
  config = ml_collections
  config.medium_type = args.medium_type
  config.device_id = args.device_id
  config.save_data = args.save_data
  config.data_path = os.path.join(
    wavebench_dataset_path, f"time_varying/{args.initial_pressure_type}")

  config.initial_pressure_type = args.initial_pressure_type
  config.domain_sidelen = 128
  config.domain_dx = 8
  # the above seetings give a domain of 1024 km x 1024 km

  config.medium_source_loc = (50, 55)
  config.medium_density = 2650
  config.pml_size = 2

  #  define properties of the propagation medium
  min_wavespeed = 1400 # [m/s]
  max_wavespeed = 4000 # [m/s]


  if config.medium_type == 'gaussian_lens':
    point_mass_strength = -31000
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

  config.source_list = sorted(absolute_file_paths(config.data_path))#[:10]


  generate_is(config)

if __name__ == '__main__':
  main()
