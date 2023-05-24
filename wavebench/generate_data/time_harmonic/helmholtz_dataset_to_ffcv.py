""" This file is used to generate the datasets in a format
compatible with the dataloaders of FFCV.

FFCV is a library that increases data
throughput in model training: https://ffcv.io/
"""

import os
import numpy as np
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField
import torch

# from wavebench.dataloaders.helmholtz_loader import HelmholtzDataset
from wavebench import wavebench_dataset_path
from wavebench.utils import absolute_file_paths, flatten_list, get_files_with_extension


helmholtz_dataset_path = os.path.join(
  wavebench_dataset_path, "time_harmonic/")


class HelmholtzDataset(torch.utils.data.Dataset):
  """Helmholtz dataset."""
  def __init__(self, kernel_type, frequency):

    if kernel_type not in ['isotropic', 'anisotropic']:
      raise AssertionError(
        'kernel_type must be either isotropic or anisotropic')

    wavespeed_paths = [absolute_file_paths(a) for a in absolute_file_paths(
      f'{helmholtz_dataset_path}/{kernel_type}') \
        if a.split('/')[-1].startswith('cp_')]

    wavespeed_paths = flatten_list(wavespeed_paths)

    self.wavespeed_paths = sorted(
      wavespeed_paths, key=lambda k: k.split('/')[-1])

    pressure_paths = []

    for path in absolute_file_paths(f'{helmholtz_dataset_path}/{kernel_type}'):
      if path.split('/')[-1].startswith('data-set_wavefield'):
        pressure_paths.append(
            get_files_with_extension(path, 'H@'))

    if frequency == 10:
      pressure_with_frequency = [a for a in flatten_list(pressure_paths)\
        if '1.00000E+01Hz' in a.split('/')[-1]]
    elif frequency == 15:
      pressure_with_frequency = [a for a in flatten_list(pressure_paths)\
        if '1.50000E+01Hz' in a.split('/')[-1]]
    elif frequency == 20:
      pressure_with_frequency = [a for a in flatten_list(pressure_paths)\
        if '2.00000E+01Hz' in a.split('/')[-1]]
    elif frequency == 40:
      pressure_with_frequency = [a for a in flatten_list(pressure_paths)\
        if '4.00000E+01Hz' in a.split('/')[-1]]
    else:
      raise NotImplementedError
    self.pressure_with_frequency = sorted(
      pressure_with_frequency, key=lambda k: k.split('/')[-3])
  def __len__(self):
    return len(self.wavespeed_paths)

  def __getitem__(self, idx, verbose_path=False):

    wavespeed = 1e-3 * np.fromfile(
        self.wavespeed_paths[idx],
        dtype=np.float32).reshape(1, 128, 128)

    if 'real' in self.pressure_with_frequency[2 * idx]:
      real_idx = 2 * idx
      img_idx = 2 * idx + 1
    else:
      real_idx = 2 * idx + 1
      img_idx = 2 * idx

    pressure_real = np.fromfile(
      self.pressure_with_frequency[real_idx],
      dtype=np.float32).reshape(1, 128, 128)

    pressure_img = np.fromfile(
      self.pressure_with_frequency[img_idx],
      dtype=np.float32).reshape(1, 128, 128)

    pressure = np.concatenate([pressure_real, pressure_img], axis=0)

    if verbose_path:
      print(f'wavespeed path {self.wavespeed_paths[idx]}')
      print(f'real pressure path {self.pressure_with_frequency[real_idx]}')
      print(f'complex pressure path {self.pressure_with_frequency[img_idx]}')
    return wavespeed, pressure



for kernel_type_ in ['isotropic', 'anisotropic']:
  for frequency_ in [10, 15, 20, 40]:
    print(kernel_type_, frequency_)
    dataset = HelmholtzDataset(
      kernel_type=kernel_type_,
      frequency=frequency_)

    write_path = f'{helmholtz_dataset_path}/{kernel_type_}_{frequency_}.beton'
    writer = DatasetWriter(write_path, {
        'input': NDArrayField(shape=(1, 128, 128), dtype=np.dtype('float32')),
        'target': NDArrayField(shape=(2, 128, 128), dtype=np.dtype('float32')),
        }, num_workers=12)

    writer.from_indexed_dataset(dataset)



