# %%
"""Convert Helmholtz dataset into the format of FFCV."""
import os
import numpy as np
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField

from wavebench.dataloaders.helmholtz_loader import HelmholtzDataset
from wavebench import wavebench_dataset_path

helmholtz_dataset_path = os.path.join(
  wavebench_dataset_path, "time_harmonic/")


for kernel_type in ['isotropic', 'anisotropic']:
  for frequency in [10, 15, 20, 40]:
    print(kernel_type, frequency)
    dataset = HelmholtzDataset(
      kernel_type=kernel_type,
      frequency=frequency)

    write_path = f'{helmholtz_dataset_path}/{kernel_type}_{frequency}.beton'
    writer = DatasetWriter(write_path, {
        'input': NDArrayField(shape=(1, 128, 128), dtype=np.dtype('float32')),
        'target': NDArrayField(shape=(2, 128, 128), dtype=np.dtype('float32')),
        }, num_workers=12)

    writer.from_indexed_dataset(dataset)



