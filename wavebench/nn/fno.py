"""Fourier Neural Operator in 2D."""
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from einops import repeat

class Conv1x1(nn.Module):
  """1x1 convolution with a custum initialization scheme."""
  def __init__(self, num_in_channels, num_out_channels, bias=True,
               init_scale = 6):
    super(Conv1x1, self).__init__()
    self.conv1x1 = nn.Conv2d(
        num_in_channels,
        num_out_channels,
        kernel_size=1,
        padding=0,
        bias=bias)

    nn.init.uniform_(
        self.conv1x1.weight,
        -np.sqrt(init_scale / num_in_channels),
        np.sqrt(init_scale / num_in_channels))

  def forward(self, x):
    return self.conv1x1(x)


class SpectralConv2d(nn.Module):
  """
  2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
  """
  def __init__(self, in_channels, out_channels, modes1, modes2):
    super(SpectralConv2d, self).__init__()


    self.in_channels = in_channels
    self.out_channels = out_channels

    # Number of Fourier modes to multiply, at most floor(N/2) + 1
    self.modes1 = modes1
    self.modes2 = modes2
    self.scale = 1 / (in_channels * out_channels)
    self.weights1 = nn.Parameter(
        self.scale * torch.rand(
            in_channels,
            out_channels,
            self.modes1,
            self.modes2,
            dtype=torch.cfloat))

    self.weights2 = nn.Parameter(
        self.scale * torch.rand(
            in_channels,
            out_channels,
            self.modes1,
            self.modes2,
            dtype=torch.cfloat))

  def forward(self, x):
    batchsize = x.shape[0]
    # Compute Fourier coeffcients up to factor of e^(- something constant)
    x_ft = torch.fft.rfft2(x)

    # Multiply relevant Fourier modes
    out_ft = torch.zeros(
        batchsize,
        self.out_channels,
        x.size(-2),
        x.size(-1)//2 + 1,
        dtype=torch.cfloat,
        device=x.device)

    out_ft[:, :, :self.modes1, :self.modes2] = \
        self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

    out_ft[:, :, -self.modes1:, :self.modes2] = \
        self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

    # Return to physical space
    x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
    return x

  def compl_mul2d(self, input, weights):
    """Complex multiplication. """
    # (B, C_in, x, y), (C_in, C_out, x,y) -> (B, C_out, x, y)
    return torch.einsum("bixy, ioxy -> boxy", input, weights)


class FourierLayer2d(nn.Module):
  """A FNO2d layer."""
  def __init__(self, num_in_channels, num_out_channels, modes1, modes2):
    super(FourierLayer2d, self).__init__()
    self.num_in_channels = num_in_channels
    self.num_out_channels = num_out_channels
    self.modes1 = modes1
    self.modes2 = modes2
    self.global_conv = SpectralConv2d(
        self.num_in_channels,
        self.num_out_channels,
        self.modes1,
        self.modes2)
    self.pointwise_conv = Conv1x1(
        self.num_in_channels,
        self.num_out_channels)

  def forward(self, x):
    x1 = self.global_conv(x)
    x2 = self.pointwise_conv(x)
    x = x1 + x2
    x = F.gelu(x)
    return x


class FNO2d(nn.Module):
  """ FNO2d network. """
  def __init__(self,
               modes1,
               modes2,
               hidden_width,
               num_in_channels=1,
               num_out_channels=1,
               lifting_channels=128,
               projection_channels=128,
               num_hidden_layer = 4):
    super(FNO2d, self).__init__()

    self.num_in_channels = num_in_channels
    self.num_out_channels = num_out_channels
    self.lift_channels = lifting_channels
    self.proj_channels = projection_channels
    self.modes1 = modes1
    self.modes2 = modes2
    self.hidden_width = hidden_width
    self.padding = 9 # pad the domain if input is non-periodic

    self.lifter = nn.Sequential(
      Conv1x1(num_in_channels+2, lifting_channels),
      nn.GELU(),
      Conv1x1(lifting_channels, hidden_width),
    )

    hidden_layers = []
    for _ in range(num_hidden_layer):
      hidden_layers.append(
        FourierLayer2d(self.hidden_width, self.hidden_width, modes1, modes2))
    self.hidden_layers = nn.Sequential(*hidden_layers)


    self.projector = nn.Sequential(
      Conv1x1(hidden_width, projection_channels),
      nn.GELU(),
      Conv1x1(projection_channels, num_out_channels),
    )

  def forward(self, x):
    x = self.concat_coordinates(x)
    x = self.lifter(x)
    x = F.pad(x, [0,self.padding, 0,self.padding])

    x = self.hidden_layers(x)

    x = x[..., :-self.padding, :-self.padding]
    x = self.projector(x)
    return x

  def concat_coordinates(self, x):
    batchsize, _, size_vert, size_horiz = x.shape
    tensors = (torch.linspace(-1, 1, steps=size_vert),
                torch.linspace(-1, 1, steps=size_horiz))
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=0)
    mgrid = repeat(mgrid, 'c x y -> b c x y', b=batchsize).to(x.device)
    encoded = torch.cat((x, mgrid), dim=1)
    return encoded
