"""Generate thick lines"""
import os
import numpy as np
import imageio
from wavebench import wavebench_dataset_path
from tqdm import tqdm

data_path = os.path.join(wavebench_dataset_path, "time_varying/thick_lines")

if not os.path.exists(data_path):
  os.makedirs(data_path)

print(f'saved to {data_path}')

def generate_random_lines_dataset(num_data):
  """Generate thick lines"""
  n = 512
  n_seg_min = 1
  n_seg_max = 5
  min_length = 20
  max_length = 100
  max_width = 12
  min_width = 6
  min_amp = .2

  for k in tqdm(range(num_data)):
    n_range = np.arange(n)
    xv, yv = np.meshgrid(n_range, n_range)
    gp = np.vstack((xv.reshape(1, n**2), yv.reshape(1, n**2)))
    n_seg = int(np.random.rand()*(n_seg_max-n_seg_min)) + n_seg_min

    f = np.zeros((n, n))

    for _ in range(n_seg):
      center = np.floor(np.random.rand(2, 1) * n)
      angle = np.random.rand(1) * np.pi
      unit_dir = np.array([np.cos(angle), np.sin(angle)])
      unit_ort = np.array([-np.sin(angle), np.cos(angle)])
      length = np.random.rand(1) * (max_length - min_length) + min_length
      width = np.random.rand(1) * (max_width - min_width) + min_width
      xi1 = np.dot(unit_dir.T, gp - center)
      xi2 = np.dot(unit_ort.T, gp - center)
      idx = (np.abs(xi1) <= length / 2) & (np.abs(xi2) <= width / 2)
      amp = np.random.rand(1) * (1 - min_amp) + min_amp
      f += amp * idx.astype(float).reshape((n, n))

    imageio.imwrite(
        os.path.join(f'{data_path}/{k}.png'), (f * 255).astype(np.uint8))
  return None

if __name__ == '__main__':
  generate_random_lines_dataset(3000)
