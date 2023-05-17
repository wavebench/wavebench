"""Reverse time continuation dataset"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
import numpy as np
from einops import rearrange
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor


from wavebench import wavebench_dataset_path
# from wavebench.utils import UnitGaussianNormalizer

rtc_dataset = os.path.join(wavebench_dataset_path, "time_varying/rtc")


class RtcDataset(Dataset):
  """The reverse time continuation dataset

  Args:
      dataset_name (str): can be `thick_lines` or `mnist`.
          Default to `thick_lines`.
      medium_type (str): can be `gaussian_lens` or `gaussian_random_field`.
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
      if medium_type in ['gaussian_lens', 'gaussian_random_field']:
        initial_pressure_dataset = np.memmap(
          f'{rtc_dataset}/{dataset_name}_{medium_type}_initial_pressure_dataset.npy', mode='r',
          shape=(5000, 128, 128), dtype=np.float32)
        final_pressure_dataset = np.memmap(
            f'{rtc_dataset}/{dataset_name}_{medium_type}_final_pressure_dataset.npy', mode='r',
            shape=(5000, 128, 128), dtype=np.float32)
      else:
        raise ValueError(f'medium_type {medium_type} not recognized.')
    elif dataset_name == 'mnist':
      # raise ValueError('mnist is not supported yet')
      if medium_type in ['gaussian_lens', 'gaussian_random_field']:
        initial_pressure_dataset = np.memmap(
          f'{rtc_dataset}/{dataset_name}_{medium_type}_initial_pressure_dataset.npy', mode='r',
          shape=(50, 128, 128), dtype=np.float32)
        final_pressure_dataset = np.memmap(
            f'{rtc_dataset}/{dataset_name}_{medium_type}_final_pressure_dataset.npy', mode='r',
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
      source = torch.from_numpy(initial_pressure_dataset).type(torch.FloatTensor)
      final = torch.from_numpy(final_pressure_dataset).type(torch.FloatTensor)

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


def get_dataloaders_rtc_thick_lines(
      medium_type='gaussian_lens',
      train_batch_size=1,
      eval_batch_size=1,
      num_train_samples=4000,
      num_val_samples=500,
      num_test_samples=500,
      resize_sidelen=None,
      num_workers=1,
      use_ffcv=False):
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
      resize_sidelen (int or None, optional): If sidelen is an integer,
          the images will be interpolated to the desinated sidelen.
          Default to None.
      num_workers (int, optional): number of workders. Defaults to 1.

  Returns:
      dataloaders: a dictionary of dataloaders for training,
          evaluation, and testing
  """
  dataset = RtcDataset(
      dataset_name='thick_lines',
      medium_type=medium_type,
      resize_sidelen=resize_sidelen,
      )

  assert num_train_samples + num_val_samples + num_test_samples <= len(dataset)

  subsets = torch.utils.data.random_split(
      dataset, [num_train_samples, num_val_samples, num_test_samples],
      generator=torch.Generator().manual_seed(42))

  image_datasets = {
      'train': subsets[0],
      'val': subsets[1],
      'test': subsets[2]}

  batch_sizes = {
      'train': train_batch_size,
      'val': eval_batch_size,
      'test': eval_batch_size
      }


  if use_ffcv:
    dataloaders = {
      x: Loader(
        f'{rtc_dataset}/thick_lines_{medium_type}.beton',
        batch_size=batch_sizes[x],
        num_workers=num_workers,
        order=OrderOption.RANDOM if x == 'train' else OrderOption.SEQUENTIAL,
        indices=image_datasets[x].indices,
        pipelines={
            'input': [NDArrayDecoder(), ToTensor()],
            'target': [NDArrayDecoder(), ToTensor()]},
        ) for x in ['train', 'val', 'test']}
  else:
    dataloaders = {
      x: DataLoader(
          image_datasets[x], batch_size=batch_sizes[x],
          shuffle=(x == 'train'), pin_memory=True,
          num_workers=num_workers) for x in ['train', 'val', 'test']}
  return dataloaders


def get_dataloaders_rtc_mnist(
        medium_type='gaussian_lens',
        resize_sidelen=None,
        batch_size=1,
        num_workers=1,
        use_ffcv=False):
  """Prepare loaders of the mnist reverse time continuation dataset.

  Args:
      medium_type (str): can be `gaussian_lens` or `gaussian_random_field`.
      resize_sidelen (int or None, optional): If sidelen is an integer,
          the images will be interpolated to the desinated sidelen.
          Default to None.
      batch_size (int, optional): batch size. Defaults to 1.
      num_workers (int, optional): number of workers. Defaults to 1.
  Returns:
      loader: the data loader.
  """
  dataset = RtcDataset(
      dataset_name='mnist',
      medium_type=medium_type,
      resize_sidelen=resize_sidelen,
      )

  if use_ffcv:
    loader = Loader(
      f'{rtc_dataset}/mnist_{medium_type}.beton',
      batch_size=1,
      num_workers=2, order=OrderOption.SEQUENTIAL,
      pipelines={
          'input': [NDArrayDecoder(), ToTensor()],
          'target': [NDArrayDecoder(), ToTensor()],
      },
      )
  else:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, pin_memory=True,
        num_workers=num_workers)

  return loader
