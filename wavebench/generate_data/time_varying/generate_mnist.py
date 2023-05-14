"""Save a few MNIST test data as png.
These MNIST images are used as OOD initial pressure for the
time-varying dataset.
"""

import os
import numpy as np
import imageio
from tqdm import tqdm
from torchvision import datasets

from wavebench import wavebench_dataset_path

mnist_data_path = os.path.join(wavebench_dataset_path, "time_varying/mnist")
num_data = 50

if not os.path.exists(mnist_data_path):
  # Save a few MNIST test data as pngs
  os.makedirs(mnist_data_path)
  mnist_test_dataset = datasets.MNIST(
    wavebench_dataset_path, train=False, download=True)

  for k in tqdm(range(num_data)):
    f = np.array(mnist_test_dataset.data[k])
    imageio.imwrite(
        os.path.join(f'{mnist_data_path}/{k}.png'), f)
