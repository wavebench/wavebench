""" WaveBench: A Benchmark Suite for Wave-based Imaging Problems"""
import os
from wavebench.utils import get_project_root

wavebench_path = str(get_project_root())

wavebench_dataset_path = os.path.join(
  wavebench_path, 'wavebench_dataset')

wavebench_checkpoint_path = os.path.join(
  wavebench_path, 'saved_models')

wavebench_figure_path = os.path.join(
  wavebench_path, 'saved_figs')
