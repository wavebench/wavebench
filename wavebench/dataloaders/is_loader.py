"""Reverse time continuation dataset"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
import numpy as np
from einops import rearrange
from wavebench import wavebench_dataset_path
import torch.nn.functional as F

is_dataset_dir = os.path.join(wavebench_dataset_path, "time_varying/is/")


class IsDataset(Dataset):
  """The reverse time continuation dataset

  Args:
      dataset_name (str): can be `thick_lines` or `mnist`.
          Default to `thick_lines`.
      sidelen: the side length of the input and target images.
          Default to 128. For lengths other than 128, the images will be
          interpolated to the desinated sidelen.
  """
  def __init__(self,
               dataset_name='thick_lines',
               sidelen=128):
    super(IsDataset, self).__init__()

    self.sidelen = sidelen

    if dataset_name == 'thick_lines':
      initial_pressure_dataset = np.memmap(
        f'{is_dataset_dir}/initial_pressure_dataset.npy', mode='r',
        shape=(3000, 512, 512), dtype=np.float32)
      boundary_measurement_dataset = np.memmap(
          f'{is_dataset_dir}/boundary_measurement_dataset.npy', mode='r',
          shape=(3000, 1334, 512), dtype=np.float32)
    elif dataset_name == 'mnist':
      raise ValueError('mnist is not supported yet')
    else:
      raise ValueError('dataset name can be either thick_lines or mnist')

    initial_pressure_dataset = np.array(initial_pressure_dataset)
    boundary_measurement_dataset = np.array(boundary_measurement_dataset)

    measurements = torch.from_numpy(initial_pressure_dataset).type(torch.FloatTensor)
    final = torch.from_numpy(boundary_measurement_dataset).type(torch.FloatTensor)

    measurements = rearrange(measurements, 'n h w -> n 1 h w')

    final = rearrange(final, 'n h w -> n 1 h w')
    final = F.interpolate(final, (512,512))

    if sidelen != 128:
      measurements = interpolate(measurements, size=[sidelen, sidelen])
      final = interpolate(final, size=[sidelen, sidelen])

    self.measurements = measurements
    self.final = final
    self.len = measurements.shape[0]

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    final_sample = self.final[idx]
    measurement_sample = self.measurements[idx]
    # [1 h w]
    return final_sample, measurement_sample


def get_dataloaders_is_thick_lines(
      train_batch_size=1,
      val_batch_size=1,
      test_batch_size=1,
      train_fraction=0.1,
      val_fraction=0.02,
      sidelen=128,
      num_workers=1):
  """Prepare loaders of the thick line reverse time continuation dataset.

  Args:
      train_batch_size (int, optional): batch size of training.
          Defaults to 1.
      val_batch_size (int, optional): batch size of evaluation.
          Defaults to 1.
      test_batch_size (int, optional): batch size of testing.
          Defaults to 1.
      train_fraction (float, optional): fraction of data for training.
          Defaults to 0.1.
      val_fraction (float, optional): fraction of data for evaluation.
          Defaults to 0.02.
      sidelen (int, optional): side length of the data. Defaults to 128.
      num_workers (int, optional): number of workders. Defaults to 1.

  Returns:
      dataloaders: a dictionary of dataloaders for training,
          evaluation, and testing
  """
  dataset = IsDataset(
      dataset_name='thick_lines',
      sidelen=sidelen,
      )

  test_fraction = 1 - train_fraction - val_fraction

  subsets = torch.utils.data.random_split(
      dataset, [train_fraction, val_fraction, test_fraction],
      generator=torch.Generator().manual_seed(42))

  image_datasets = {
      'train': subsets[0],
      'val': subsets[1],
      'test': subsets[2]}

  batch_sizes = {
      'train': train_batch_size,
      'val': val_batch_size,
      'test': test_batch_size}

  dataloaders = {
      x: DataLoader(
          image_datasets[x], batch_size=batch_sizes[x],
          shuffle=(x == 'train'), pin_memory=True,
          num_workers=num_workers) for x in ['train', 'val', 'test']}
  return dataloaders

