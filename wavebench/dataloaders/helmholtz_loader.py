"""Helmholtz dataset"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor
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



# f'{wavebench_dataset_path}/time_harmonic/{kernel_type}_{frequency}.beton'
def get_dataloaders_helmholtz(
      kernel_type='isotropic',
      frequency=10,
      train_batch_size=1,
      eval_batch_size=1,
      num_train_samples=49000,
      num_val_samples=500,
      num_test_samples=500,
      sidelen=None,
      num_workers=1,
      use_ffcv=False):
  """Prepare loaders of the Helmholtz dataset.

  Args:
      kernel_type: can be `isotropic` or `anisotropic`.
      frequency: can be 10, 15, 20, 40 [Hz].
      train_batch_size (int, optional): batch size of training.
          Defaults to 1.
      test_batch_size (int, optional): batch size of testing.
          Defaults to 1.
      train_fraction (float, optional): fraction of data for training.
          Defaults to 0.625.
      test_fraction (float, optional): fraction of data for testing.
          Defaults to 0.125.
      sidelen: the side length of the input and target images.
          Default to None (keeps the original size).
          For an integer-valued sidelen, the images will be
          interpolated to the desinated sidelen.
      num_workers (int, optional): number of workers for data loading.
  """

  sum_samples = num_train_samples + num_val_samples + num_test_samples
  assert sum_samples <= 50000

  batch_sizes = {
      'train': train_batch_size,
      'val': eval_batch_size,
      'test': eval_batch_size
      }

  if use_ffcv:
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(sum_samples, generator=generator).tolist()

    splitted_indices = {
      'train': indices[:num_train_samples],
      'val': indices[num_train_samples:num_train_samples+num_val_samples],
      'test': indices[num_train_samples+num_val_samples:]
    }

    dataloaders = {
      x: Loader(
        f'{wavebench_dataset_path}/time_harmonic/{kernel_type}_{frequency}.beton',
        batch_size=batch_sizes[x],
        num_workers=num_workers,
        order=OrderOption.RANDOM if x == 'train' else OrderOption.SEQUENTIAL,
        indices=splitted_indices[x],
        pipelines={
            'input': [NDArrayDecoder(), ToTensor()],
            'target': [NDArrayDecoder(), ToTensor()]},
        ) for x in ['train', 'val', 'test']}
  else:
    dataset = HelmholtzDataset(
        kernel_type=kernel_type,
        frequency=frequency,
        )

    subsets = torch.utils.data.random_split(
        dataset, [num_train_samples, num_val_samples, num_test_samples],
        generator=torch.Generator().manual_seed(42))

    image_datasets = {
        'train': subsets[0],
        'val': subsets[1],
        'test': subsets[2]}

    dataloaders = {
      x: DataLoader(
          image_datasets[x], batch_size=batch_sizes[x],
          shuffle=(x == 'train'), pin_memory=True,
          num_workers=num_workers) for x in ['train', 'val', 'test']}
  return dataloaders

