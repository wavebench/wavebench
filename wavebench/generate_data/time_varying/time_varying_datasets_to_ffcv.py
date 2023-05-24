""" This file is used to generate the datasets in a format
compatible with the dataloaders of FFCV.

FFCV is a library that increases data
throughput in model training: https://ffcv.io/
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from einops import rearrange
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField
from wavebench import wavebench_dataset_path


rtc_dataset = os.path.join(wavebench_dataset_path, "time_varying/rtc")
is_dataset_dir = os.path.join(wavebench_dataset_path, "time_varying/is/")


class IsDataset(Dataset):
  """The reverse time continuation dataset

  Args:
      dataset_name (str): can be `thick_lines` or `mnist`.
          Default to `thick_lines`.
      medium_type (str): can be `gaussian_lens`, `grf_isotropic`,
          or 'grf_anisotropic`.
      resize_sidelen: the side length of the input and target images.
          Default to None. If sidelen is an integer, the images will be
          interpolated to the desinated sidelen.
  """
  def __init__(self,
               dataset_name='thick_lines',
               medium_type='gaussian_lens',
               resize_sidelen=None,
               numpy=False):
    super(IsDataset, self).__init__()

    if dataset_name == 'thick_lines':
      if medium_type in ['gaussian_lens', 'grf_isotropic', 'grf_anisotropic']:
        initial_pressure_dataset = np.memmap(
            f'{is_dataset_dir}/{dataset_name}_{medium_type}' +\
              '_initial_pressure_dataset.npy',
            mode='r', shape=(10000, 128, 128), dtype=np.float32)
        boundary_measurement_dataset = np.memmap(
            f'{is_dataset_dir}/{dataset_name}_{medium_type}' +\
              '_boundary_measurement_dataset.npy',
            mode='r', shape=(10000, 334, 128), dtype=np.float32)
      else:
        raise ValueError(f'medium_type {medium_type} not recognized.')
    elif dataset_name == 'mnist':
      if medium_type in ['gaussian_lens', 'grf_isotropic', 'grf_anisotropic']:
        initial_pressure_dataset = np.memmap(
          f'{is_dataset_dir}/{dataset_name}_{medium_type}' +\
            '_initial_pressure_dataset.npy', mode='r',
          shape=(50, 128, 128), dtype=np.float32)
        boundary_measurement_dataset = np.memmap(
            f'{is_dataset_dir}/{dataset_name}_{medium_type}' +\
              '_boundary_measurement_dataset.npy', mode='r',
            shape=(50, 334, 128), dtype=np.float32)
      else:
        raise ValueError(f'medium_type {medium_type} not recognized.')
    else:
      raise ValueError('dataset name can be either thick_lines or mnist')

    initial_pressure_dataset = np.array(initial_pressure_dataset)
    boundary_measurement_dataset = np.array(boundary_measurement_dataset)

    initial = torch.from_numpy(
        initial_pressure_dataset).type(torch.FloatTensor)
    measurements = torch.from_numpy(
        boundary_measurement_dataset).type(torch.FloatTensor)

    initial = rearrange(initial, 'n h w -> n 1 h w')
    measurements = rearrange(measurements, 'n h w -> n 1 h w')

    if resize_sidelen is not None:
      print(f'interpolating images to size {resize_sidelen}')
      initial = interpolate(
          initial, size=[resize_sidelen, resize_sidelen],
          mode='bicubic')

    measurements = interpolate(
        measurements,
        size=[initial.shape[-1], initial.shape[-2]],
        mode='nearest')

    if numpy:
      initial = np.array(initial).astype('float32')
      measurements = np.array(measurements).astype('float32')
    self.measurements = measurements
    self.initial = initial
    self.len = measurements.shape[0]

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    measurement_sample = self.measurements[idx]
    initial_sample = self.initial[idx]

    # [1 h w]
    return measurement_sample, initial_sample


class RtcDataset(Dataset):
  """The reverse time continuation dataset

  Args:
      dataset_name (str): can be `thick_lines` or `mnist`.
          Default to `thick_lines`.
      medium_type (str): can be `gaussian_lens`, `grf_isotropic`,
        or 'grf_anisotropic`.
      resize_sidelen: the side length of the input and target images.
          Default to None. If sidelen is an integer, the images will be
          interpolated to the desinated sidelen.
  """
  def __init__(self,
               dataset_name='thick_lines',
               medium_type='gaussian_lens',
               resize_sidelen=None,
               numpy=False):
    super(RtcDataset, self).__init__()

    if dataset_name == 'thick_lines':
      if medium_type in ['gaussian_lens', 'grf_isotropic', 'grf_anisotropic']:
        initial_pressure_dataset = np.memmap(
          f'{rtc_dataset}/{dataset_name}_{medium_type}' +\
            '_initial_pressure_dataset.npy', mode='r',
          shape=(10000, 128, 128), dtype=np.float32)
        final_pressure_dataset = np.memmap(
            f'{rtc_dataset}/{dataset_name}_{medium_type}' +\
              '_final_pressure_dataset.npy', mode='r',
            shape=(10000, 128, 128), dtype=np.float32)
      else:
        raise ValueError(f'medium_type {medium_type} not recognized.')
    elif dataset_name == 'mnist':
      if medium_type in ['gaussian_lens', 'grf_isotropic', 'grf_anisotropic']:
        initial_pressure_dataset = np.memmap(
          f'{rtc_dataset}/{dataset_name}_{medium_type}' +\
            '_initial_pressure_dataset.npy', mode='r',
          shape=(50, 128, 128), dtype=np.float32)
        final_pressure_dataset = np.memmap(
            f'{rtc_dataset}/{dataset_name}_{medium_type}' +\
              '_final_pressure_dataset.npy', mode='r',
            shape=(50, 128, 128), dtype=np.float32)
      else:
        raise ValueError(f'medium_type {medium_type} not recognized.')
    else:
      raise ValueError('dataset name can be either thick_lines or mnist')

    initial_pressure_dataset = np.array(initial_pressure_dataset)
    final_pressure_dataset = np.array(final_pressure_dataset)

    if numpy:
      source = initial_pressure_dataset.astype('float32')
      final = final_pressure_dataset.astype('float32')
    else:
      source = torch.from_numpy(initial_pressure_dataset).type(
        torch.FloatTensor)
      final = torch.from_numpy(final_pressure_dataset).type(
        torch.FloatTensor)

    source = rearrange(source, 'n h w -> n 1 h w')
    final = rearrange(final, 'n h w -> n 1 h w')

    if resize_sidelen is not None:
      print(f'interpolating images to size {resize_sidelen}')
      source = interpolate(source, size=[resize_sidelen, resize_sidelen],
                          mode='bicubic')
      final = interpolate(final, size=[resize_sidelen, resize_sidelen],
                          mode='bicubic')

    self.source = source
    self.final = final
    self.len = source.shape[0]

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    final_sample = self.final[idx]
    source_sample = self.source[idx]
    # [1 h w]
    return final_sample, source_sample


if __name__ == '__main__':
  for problem in ['rtc', 'is']:
    for dataset_name_ in ['thick_lines', 'mnist']:
      for medium_type_ in ['gaussian_lens', 'grf_anisotropic', 'grf_isotropic']:
        print(problem, dataset_name_, medium_type_)
        if problem == 'rtc':
          dataset_path = os.path.join(
            wavebench_dataset_path, "time_varying/rtc")
          dataset = RtcDataset(
            dataset_name=dataset_name_,
            medium_type=medium_type_,
            numpy=True)
        elif problem == 'is':
          dataset_path = os.path.join(wavebench_dataset_path, "time_varying/is")
          dataset = IsDataset(
            dataset_name=dataset_name_,
            medium_type=medium_type_,
            numpy=True)

        write_path = f'{dataset_path}/{dataset_name_}_{medium_type_}.beton'
        writer = DatasetWriter(write_path, {
            'input': NDArrayField(
              shape=(1, 128, 128), dtype=np.dtype('float32')),
            'target': NDArrayField(
              shape=(1, 128, 128), dtype=np.dtype('float32')),
            }, num_workers=12)

        writer.from_indexed_dataset(dataset)


