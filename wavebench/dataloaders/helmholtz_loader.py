"""Helmholtz dataset"""
import os
import torch
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor
from wavebench import wavebench_dataset_path

helmholtz_dataset_path = os.path.join(
  wavebench_dataset_path, "time_harmonic/")

def get_dataloaders_helmholtz(
      kernel_type='isotropic',
      frequency=10,
      train_batch_size=1,
      eval_batch_size=1,
      num_train_samples=49000,
      num_val_samples=500,
      num_test_samples=500,
      num_workers=1,
      is_elastic=False
      ):
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
      num_workers (int, optional): number of workers for data loading.
  """

  sum_samples = num_train_samples + num_val_samples + num_test_samples
  assert sum_samples <= 50000

  batch_sizes = {
      'train': train_batch_size,
      'val': eval_batch_size,
      'test': eval_batch_size
      }

  generator = torch.Generator().manual_seed(42)
  indices = torch.randperm(sum_samples, generator=generator).tolist()

  splitted_indices = {
    'train': indices[:num_train_samples],
    'val': indices[num_train_samples:num_train_samples+num_val_samples],
    'test': indices[num_train_samples+num_val_samples:]
  }

  if is_elastic:
    wavetype = 'elastic'
    if kernel_type != 'anisotropic':
      raise ValueError('Elastic kernel_type must be anisotropic')
  else:
    wavetype = 'acoustic'

  dataloaders = {
    x: Loader(
      f'{wavebench_dataset_path}/time_harmonic/{wavetype}/{kernel_type}_{wavetype}_{int(frequency)}.beton',
      batch_size=batch_sizes[x],
      num_workers=num_workers,
      order=OrderOption.RANDOM if x == 'train' else OrderOption.SEQUENTIAL,
      indices=splitted_indices[x],
      pipelines={
          'input': [NDArrayDecoder(), ToTensor()],
          'target': [NDArrayDecoder(), ToTensor()]},
      ) for x in ['train', 'val', 'test']}

  return dataloaders

