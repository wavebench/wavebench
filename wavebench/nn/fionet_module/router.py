'''
The Siren network modified from https://github.com/lucidrains/siren-pytorch
'''
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
import torchfields  # pylint: disable=unused-import # noqa: F401


# helpers
def exists(val):  # pylint: disable=missing-function-docstring
  return val is not None


def cast_tuple(val, num_repeat=1):
  # pylint: disable=missing-function-docstring
  return val if isinstance(val, tuple) else ((val,) * num_repeat)


# sin activation
class Sine(nn.Module):
  """ Sine activation function """
  def __init__(self, w0=1.):
    super().__init__()
    self.w0 = w0  # pylint: disable=invalid-name

  def forward(self, x):
    # pylint: disable=missing-function-docstring, invalid-name
    return torch.sin(self.w0 * x)


# siren layer
class Siren(nn.Module):
  """ Siren layer """
  def __init__(self, dim_in, dim_out, w0=1., c=6.,
              is_first=False, use_bias=True, activation=None):
    super().__init__()
    self.dim_in = dim_in
    self.is_first = is_first

    weight = torch.zeros(dim_out, dim_in)
    bias = torch.zeros(dim_out) if use_bias else None
    self.init_(weight, bias, c=c, w0=w0)

    self.weight = nn.Parameter(weight)
    self.bias = nn.Parameter(bias) if use_bias else None
    self.activation = Sine(w0) if activation is None else activation

  def init_(self, weight, bias, c, w0):
    # pylint: disable=missing-function-docstring, invalid-name
    dim = self.dim_in

    w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
    weight.uniform_(-w_std, w_std)

    if exists(bias):
      bias.uniform_(-w_std, w_std)

  def forward(self, x):
    # pylint: disable=missing-function-docstring, invalid-name
    out = F.linear(x, self.weight, self.bias)
    out = self.activation(out)
    return out


# Siren network
class SirenNet(nn.Module):
  """ Siren network """
  def __init__(self, dim_in, dim_hidden, dim_out, num_layers,
                c=6., w0=1., w0_initial=30.,
                use_bias=True, final_activation=None):
    super().__init__()
    self.num_layers = num_layers
    self.dim_hidden = dim_hidden

    self.layers = nn.ModuleList([])
    for ind in range(num_layers):
      is_first = ind == 0
      layer_w0 = w0_initial if is_first else w0
      layer_dim_in = dim_in if is_first else dim_hidden

      self.layers.append(Siren(
          dim_in=layer_dim_in,
          dim_out=dim_hidden,
          w0=layer_w0,
          c=c,
          use_bias=use_bias,
          is_first=is_first
      ))

    final_activation = nn.Identity() if not exists(
        final_activation) else final_activation
    self.last_layer = Siren(
        dim_in=dim_hidden, dim_out=dim_out, w0=w0,
        use_bias=use_bias, activation=final_activation)

  def forward(self, x, mods=None):
    # pylint: disable=missing-function-docstring, invalid-name
    mods = cast_tuple(mods, self.num_layers)

    for layer, mod in zip(self.layers, mods):
      x = layer(x)

      if exists(mod):
        x *= rearrange(mod, 'd -> () d')

    return self.last_layer(x)


class SirenRouter(nn.Module):
  """ Siren router """
  def __init__(self,
              dim_in, dim_hidden,
              dim_out,
              num_layers,
              c=6., w0=1., w0_initial=30.,
              use_bias=True, final_activation=None):
    super().__init__()

    self.net = SirenNet(
        dim_in=dim_in,
        dim_hidden=dim_hidden,
        dim_out=dim_out,
        num_layers=num_layers,
        c=c,
        w0=w0, w0_initial=w0_initial,
        use_bias=use_bias,
        final_activation=final_activation)

  def get_mgrid(self, image_sidelen):
    tensors = [torch.linspace(-1, 1, steps=image_sidelen),
                torch.linspace(-1, 1, steps=image_sidelen)]
    mgrid = torch.stack(
      torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = rearrange(mgrid, 'h w c -> (h w) c')
    return mgrid

  def forward(self, latents, image_sidelen):
    '''foward pass on a multiple latent vectors
        of dim (num_latents, dim_latents)
    '''

    mgrid = self.get_mgrid(image_sidelen).to(latents.device)
    num_coords = image_sidelen * image_sidelen

    def forward_single_latent(latent):
      '''fowrad pass on a single latent vector'''
      # concat the fixed grid and the input latents
      repeated_latents = repeat(latent, 'd -> c d',
                                c=num_coords)

      coords_and_latents = torch.cat(
        (mgrid, repeated_latents), 1)

      out = self.net(coords_and_latents)

      out = rearrange(
        out, '(h w) c -> c h w',
        h=image_sidelen,
        w=image_sidelen)
      return out

    out = torch.vmap(forward_single_latent)(latents)
    return out

  def get_resampling_mapping(self, latents, image_sidelen):
    """Get the resampling mapping.
    The resampling mapping is a tensor that has the shape
    [num_curvelets, 2, sidelen, sidelen]. The value [-1, -1] corresponds
    to the top-left corner and [1, 1] corresponds to the
    bottom-right corner. Values out side of [-1, -1] x [1, 1] are out of
    bounds.
    """
    # output resampling mapping from the router
    resampling_mapping = self.forward(latents, image_sidelen).field()

    return resampling_mapping


  def warp(self, latents, x):  # pylint: disable=invalid-name
    """Warp the grid that underlies the input"""

    x = rearrange(x, '... c h w -> ... c 1 h w')
    image_sidelen = x.shape[-1]

    def get_displacement_fields(latents):
      """Get the displacement vector field.
      The displacement vector field is a tensor has the shape
      [num_curvelets, 2, sidelen, sidelen].
      """
      # convert from resampling mapping (predicted by the router)
      # to displacement_fields
      resampling_mapping = self.get_resampling_mapping(latents, image_sidelen)
      displacement_fields = resampling_mapping.from_mapping()
      return displacement_fields

    warper = torch.vmap(get_displacement_fields(latents).sample)
    out = warper(x)  # in shape [..., c, 1, h, w]
    out = rearrange(out, '... c 1 h w -> ... c h w')
    return out
