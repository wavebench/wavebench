"""Reverse time continuation dataset"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
import numpy as np
from einops import rearrange
from wavebench import wavebench_dataset_path

rtc_dataset = os.path.join(wavebench_dataset_path, "time_varying/rtc")


class RtcDataset(Dataset):
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
               medium_type='gaussian_lens',
               sidelen=128):
    super(RtcDataset, self).__init__()

    self.sidelen = sidelen

    if dataset_name == 'thick_lines':
      if medium_type in ['gaussian_lens', 'gaussian_random_field']:
        initial_pressure_dataset = np.memmap(
          f'{rtc_dataset}/{medium_type}_initial_pressure_dataset.npy', mode='r',
          shape=(3000, 512, 512), dtype=np.float32)
        final_pressure_dataset = np.memmap(
            f'{rtc_dataset}/{medium_type}_final_pressure_dataset.npy', mode='r',
            shape=(3000, 512, 512), dtype=np.float32)
      else:
        raise ValueError(f'medium_type {medium_type} not recognized.')
    elif dataset_name == 'mnist':
      raise ValueError('mnist is not supported yet')
    else:
      raise ValueError('dataset name can be either thick_lines or mnist')

    initial_pressure_dataset = np.array(initial_pressure_dataset)
    final_pressure_dataset = np.array(final_pressure_dataset)

    source = torch.from_numpy(initial_pressure_dataset).type(torch.FloatTensor)
    final = torch.from_numpy(final_pressure_dataset).type(torch.FloatTensor)

    source = rearrange(source, 'n h w -> n 1 h w')
    final = rearrange(final, 'n h w -> n 1 h w')

    source = interpolate(source, size=[sidelen, sidelen])
    final = interpolate(final, size=[sidelen, sidelen])

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


def get_dataloaders_rtc_thick_lines(
      medium_type='gaussian_lens',
      train_batch_size=1,
      val_batch_size=1,
      test_batch_size=1,
      train_fraction=0.1,
      val_fraction=0.02,
      sidelen=128,
      num_workers=1):
  """Prepare loaders of the thick line reverse time continuation dataset.

  Args:
      medium_type: can be `gaussian_lens` or `gaussian_random_field`.
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
  dataset = RtcDataset(
      dataset_name='thick_lines',
      medium_type=medium_type,
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


# def get_dataloaders_rtc_mnist(
#         sidelen=128,
#         batch_size=1,
#         num_workers=1):
#   """Prepare loaders of the mnist reverse time continuation dataset.

#   Args:
#       batch_size (int, optional): batch size. Defaults to 1.
#       num_workers (int, optional): number of workers. Defaults to 1.

#   Returns:
#       loader: the data loader.
#   """
#   dataset = RtcDataset(
#       sidelen=sidelen,
#       dataset_name='rtc_mnist',
#       )

#   loader = DataLoader(
#       dataset,
#       batch_size=batch_size,
#       shuffle=False, pin_memory=True,
#       num_workers=num_workers)
#   return loader
