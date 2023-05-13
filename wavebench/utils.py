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

