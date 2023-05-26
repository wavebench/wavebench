""" Utility functions for the WaveBench package. """
import random
import os
import glob
from pathlib import Path
import numpy as np
import torch


def get_project_root() -> Path:
  return Path(__file__).parent.parent

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

def flatten_list(l):
  return [item for sublist in l for item in sublist]

def get_files_with_extension(folder_path, extension):
  file_paths = set()  # Use a set to ensure uniqueness

  # Recursive function to search for files in subfolders
  def search_files(current_folder):
    for root, dirs, files in os.walk(current_folder):
      for file in files:
        # Check if the file has the desired extension
        if file.endswith(extension):
          # Append the normalized absolute path of the file to the set
          file_paths.add(os.path.normpath(os.path.join(root, file)))

      for directory in dirs:
        # Recursively search for files in subfolders
        search_files(os.path.join(root, directory))

  # Start searching files from the provided folder path
  search_files(folder_path)
  return file_paths