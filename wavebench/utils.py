import random
import os
import glob
import numpy as np
import torch


def seed_everything(seed: int):
  """Seed everything for reproducibility"""
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

def absolute_file_paths(directory):
  return glob.glob(os.path.join(directory, "**"))


# normalization, pointwise gaussian
class UnitGaussianNormalizer:
  def __init__(self, x, eps=0.00001, reduce_dim=[0], verbose=True):
    super().__init__()
    n_samples, *shape = x.shape
    self.sample_shape = shape
    self.verbose = verbose
    self.reduce_dim = reduce_dim

    # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
    self.mean = torch.mean(x, reduce_dim, keepdim=True).squeeze(0)
    self.std = torch.std(x, reduce_dim, keepdim=True).squeeze(0)
    self.eps = eps

    if verbose:
      print(f'UnitGaussianNormalizer init on {n_samples}, reducing over {reduce_dim}, samples of shape {shape}.')
      print(f'   Mean and std of shape {self.mean.shape}, eps={eps}')

  def encode(self, x):
    x -= self.mean
    x /= (self.std + self.eps)
    return x

  def decode(self, x, sample_idx=None):
    if sample_idx is None:
      std = self.std + self.eps # n
      mean = self.mean
    else:
      if len(self.mean.shape) == len(sample_idx[0].shape):
        std = self.std[sample_idx] + self.eps  # batch*n
        mean = self.mean[sample_idx]
      if len(self.mean.shape) > len(sample_idx[0].shape):
        std = self.std[:,sample_idx]+ self.eps # T*batch*n
        mean = self.mean[:,sample_idx]

    x *= std
    x += mean

    return x
