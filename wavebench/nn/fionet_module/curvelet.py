"""Curvelet decomposition"""
import torch
from torch import nn
from diffcurve.fdct2d.curvelet_2d import get_curvelet_system
from diffcurve.fdct2d.torch_frontend import torch_fdct_2d, torch_ifdct_2d


class CurveletDecomp(nn.Module):
  """The curvelet decomposition module"""
  def __init__(self, sidelen, is_real=0, finest=2, nbscales=4,
              nbangles_coarse=16,
              use_only_middle_scales=True,
              ):
    super(CurveletDecomp, self).__init__()
    dct_kwargs = {
        'is_real': float(is_real),
        'finest': float(finest),
        'nbscales': float(nbscales),
        'nbangles_coarse': float(nbangles_coarse)}
    self.nbangles_coarse = nbangles_coarse
    self.nbscales = nbscales
    curvelet_system, curvelet_coeff_dim = get_curvelet_system(
        sidelen, sidelen, dct_kwargs)

    curvelet_system = torch.from_numpy(curvelet_system)
    curvelet_coeff_dim = torch.from_numpy(curvelet_coeff_dim)
    curvelet_support_size = torch.prod(curvelet_coeff_dim, 1)

    self.use_only_middle_scales = use_only_middle_scales
    if use_only_middle_scales:
      # ignore the first and the last scale
      curvelet_system = curvelet_system[1:-1, ...]
      curvelet_support_size = curvelet_support_size[1:-1]

    self.sidelen = sidelen
    self.num_curvelets = curvelet_system.shape[0]
    self.register_buffer('curvelet_system',
                          torch.view_as_real(curvelet_system))
    self.register_buffer('curvelet_support_size', curvelet_support_size)

    curvelet_directions = self.get_directions_of_curvelets()
    curvelet_scales = self.get_onehot_scales_of_curvelets()
    curvelet_directions_scales = torch.cat(
        (curvelet_directions, curvelet_scales), dim=1)

    self.register_buffer('curvelet_directions',
                          curvelet_directions)
    self.register_buffer('curvelet_directions_scales',
                          curvelet_directions_scales)

  def get_onehot_scales_of_curvelets(self):
    """ Get the one-hot encoding of the scales of the curvelets """
    scales = [0]
    num_scales = self.nbangles_coarse
    for scale_idx in range(1, self.nbscales-1):
      scales += [scale_idx] * num_scales
      num_scales *= 2
    scales.append(0)

    one_hot_scales = torch.nn.functional.one_hot(
        torch.FloatTensor(scales).to(torch.int64))

    if self.use_only_middle_scales:
      one_hot_scales = one_hot_scales[1:-1, ...]
    return one_hot_scales

  def get_directions_of_curvelets(self):
    """ Get the directions of the curvelets """
    curvelet_directions_list = []

    curvelet_system = torch.view_as_complex(self.curvelet_system)
    for curvelet_in_freq in curvelet_system:
      nz_coords = torch.argwhere(
          torch.abs(curvelet_in_freq) > 1e-8).type(torch.float)

      nz_coords[:, 0] -= curvelet_in_freq.shape[0]/2

      nz_coords[:, 1] -= curvelet_in_freq.shape[1]/2

      mean_direction = nz_coords.mean(0)

      mean_direction_norm = mean_direction.norm()

      mean_direction *= (mean_direction_norm > 1e-8).unsqueeze(0)

      mean_direction_norm += (mean_direction_norm == 0) * 1e-8

      direction = mean_direction / mean_direction_norm

      curvelet_directions_list.append(direction)
    curvelet_directions = torch.stack(curvelet_directions_list)
    return curvelet_directions

  def forward(self, x):  # pylint: disable=invalid-name
    """Forward pass"""
    curvelet_system = torch.view_as_complex(self.curvelet_system)
    torch_coeff = torch_fdct_2d(x, curvelet_system)
    torch_decomp = torch_ifdct_2d(torch_coeff,
                                  curvelet_system,
                                  self.curvelet_support_size
                                  ).real.type(torch.float)
    return torch_decomp
    # output has the shape (B, num_curvelets, 1, H, W)
    # return torch_decomp.reshape((-1,
    #                              self.num_curvelets,
    #                              self.sidelen, self.sidelen))
