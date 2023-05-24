"""Measuring time for forward and backward pass.
The code is adapted from:
https://gist.github.com/iacolippo/9611c6d9c7dfc469314baeb5a69e7e1b
"""

import gc
import sys
import time
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from wavebench.nn.unet import UNet
from wavebench.nn.fno import FNO2d

def count_params(model):
  """Returns the number of parameters of a PyTorch model"""
  return sum(
    [p.numel()*2 if p.is_complex() else p.numel() for p in model.parameters()])


def measure(model, x, y):
  # synchronize gpu time and measure fp
  torch.cuda.synchronize()
  t0 = time.time()
  y_pred = model(x)
  torch.cuda.synchronize()
  elapsed_fp = time.time()-t0

  # zero gradients, synchronize time and measure
  model.zero_grad()
  t0 = time.time()
  y_pred.backward(y)
  torch.cuda.synchronize()
  elapsed_bp = time.time()-t0
  return elapsed_fp, elapsed_bp

def benchmark(model, x, y):
  # transfer the model on GPU
  model.cuda()

  # DRY RUNS
  for _ in range(10):
    _, _ = measure(model, x, y)

  print('DONE WITH DRY RUNS, NOW BENCHMARKING')

  # START BENCHMARKING
  t_forward = []
  t_backward = []
  for i in range(100):
    t_fp, t_bp = measure(model, x, y)
    t_forward.append(t_fp)
    t_backward.append(t_bp)

  # free memory
  del model

  return t_forward, t_backward


def main():
  # set the seed for RNG
  if len(sys.argv)==2:
    torch.manual_seed(int(sys.argv[1]))
  else:
    torch.manual_seed(1234)

  # set cudnn backend to benchmark config
  cudnn.benchmark = True

  # instantiate the models

  fno_depth_4 = FNO2d(modes1=16,
      modes2=16,
      hidden_width=32,
      num_hidden_layers=4,)

  fno_depth_8 = FNO2d(modes1=16,
      modes2=16,
      hidden_width=32,
      num_hidden_layers=8,)

  unet_ch_32 = UNet(
  n_input_channels=1,
  n_output_channels=1,
  channel_reduction_factor=2)

  unet_ch_64 = UNet(
  n_input_channels=1,
  n_output_channels=1,
  channel_reduction_factor=1)


  architectures = {'fno_depth_4': fno_depth_4,
              'fno_depth_8': fno_depth_8,
              'unet_ch_32': unet_ch_32,
              'unet_ch_64': unet_ch_64}

  # build dummy variables to input and output
  x = Variable(torch.randn(8, 1, 128, 128)).cuda()
  y = torch.randn(8, 1, 128, 128).cuda()

  # loop over architectures and measure them
  for model_name, model in architectures.items():
    t_fp, t_bp = benchmark(model, x, y)
    # print results
    print(f"MODEL: {model_name}")
    print(f"NUM PARAMS: {count_params(model) / 100**3} M")
    print('FORWARD PASS: ',
          np.mean(np.asarray(t_fp)), '+/-', np.std(np.asarray(t_fp)))
    print('BACKWARD PASS: ',
          np.mean(np.asarray(t_bp)), '+/-', np.std(np.asarray(t_bp)))
    print('RATIO BP/FP:',
          np.mean(np.asarray(t_bp))/np.mean(np.asarray(t_fp)))
    # force garbage collection
    gc.collect()

if __name__ == '__main__':
  main()
