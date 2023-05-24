"""Reverse time continuation dataset"""
import os
import torch
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor
from wavebench import wavebench_dataset_path

is_dataset_dir = os.path.join(wavebench_dataset_path, "time_varying/is/")


def get_dataloaders_is_thick_lines(
      medium_type='gaussian_lens',
      train_batch_size=1,
      eval_batch_size=1,
      num_train_samples=9000,
      num_val_samples=500,
      num_test_samples=500,
      num_workers=1,
      ):
  """Prepare loaders of the thick line reverse time continuation dataset.

  Args:
      medium_type: can be `gaussian_lens` or `gaussian_random_field`.
      train_batch_size (int, optional): batch size of training.
          Defaults to 1.
      eval_batch_size (int, optional): batch size of validation & testing.
          Defaults to 1.
      num_train_samples (int): number of training samples.
      num_val_samples (int): number of validation samples.
      num_test_samples (int): number of test samples.
      num_workers (int, optional): number of workders. Defaults to 1.

  Returns:
      dataloaders: a dictionary of dataloaders for training,
          evaluation, and testing
  """

  sum_samples = num_train_samples + num_val_samples + num_test_samples
  assert sum_samples <= 10000

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


  dataloaders = {
      x: Loader(
      f'{is_dataset_dir}/thick_lines_{medium_type}.beton',
      batch_size=batch_sizes[x],
      num_workers=num_workers,
      order=OrderOption.RANDOM if x == 'train' else OrderOption.SEQUENTIAL,
      indices=splitted_indices[x],
      pipelines={
          'input': [NDArrayDecoder(), ToTensor()],
          'target': [NDArrayDecoder(), ToTensor()]},
      ) for x in ['train', 'val', 'test']
      }

  return dataloaders

def get_dataloaders_is_mnist(
        medium_type='gaussian_lens',
        batch_size=1,
        num_workers=1,
        ):
  """Prepare loaders of the mnist inverse source dataset.

  Args:
      medium_type (str): can be `gaussian_lens` or `gaussian_random_field`.
      batch_size (int, optional): batch size. Defaults to 1.
      num_workers (int, optional): number of workers. Defaults to 1.
  Returns:
      loader: the data loader.
  """

  loader = Loader(
    f'{is_dataset_dir}/mnist_{medium_type}.beton',
    batch_size=1,
    num_workers=2, order=OrderOption.SEQUENTIAL,
    pipelines={
        'input': [NDArrayDecoder(), ToTensor()],
        'target': [NDArrayDecoder(), ToTensor()],
    },
    )

  return loader
