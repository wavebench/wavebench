"""FIONet"""
# import functorch
import torchfields  # pylint: disable=unused-import # noqa: F401
from torch import nn
import warnings
from einops.layers.torch import Reduce
from wavebench.nn.fionet_module.router import SirenRouter
from wavebench.nn.fionet_module.curvelet import CurveletDecomp
from wavebench.nn import unet

class FIONet(nn.Module):
  """The FioNet module"""
  def __init__(
      self,
      sidelen,
      n_output_channels=1,
      use_two_routers=False,
      router_use_curvelet_scales=False,
      keep_only_curvelet_middle_scales=True,
      siren_latent_dim=512,
      siren_num_layers=5,
      siren_omega=30.,
      siren_c=6.,
      out_op='sum',
      out_activation='id',
      unet_channel_redu_factor=2):
    super().__init__()

    if n_output_channels !=1 and out_op == 'sum':
      warnings.warn(
          'out_op is set to sum but n_output_channels is not 1. '
          'This is probably a mistake. '
          'out_op is set to conv instead.')
      out_op = 'conv'

    self.use_two_routers = use_two_routers
    self.sidelen = sidelen
    self.decomposer = CurveletDecomp(
        sidelen,
        use_only_middle_scales=keep_only_curvelet_middle_scales)
    self.num_curvelets = self.decomposer.curvelet_directions.shape[0]

    if out_activation.lower() == 'id':
      self.out_activation = nn.Identity()
    elif out_activation.lower() == 'relu':
      self.out_activation = nn.ReLU()
    else:
      raise NotImplementedError(
          'only identity and ReLU functions are implemented.')

    if router_use_curvelet_scales:
      router_input = self.decomposer.curvelet_directions_scales
    else:
      router_input = self.decomposer.curvelet_directions

    siren_dim_in = router_input.shape[1] + 2
    self.register_buffer('router_input', router_input)

    siren_params = {
        'dim_in': siren_dim_in,
        'dim_hidden': siren_latent_dim,
        'dim_out': 2,
        'num_layers': siren_num_layers,
        'w0': siren_omega,
        'c': siren_c,
        'w0_initial': siren_omega
        }

    self.router_1 = SirenRouter(
        image_sidelen=sidelen,
        **siren_params
    )

    if use_two_routers:
      self.router_2 = SirenRouter(
          image_sidelen=sidelen,
          **siren_params
      )

    self.unet = unet.UNet(
        self.num_curvelets,
        self.num_curvelets,
        channel_reduction_factor=unet_channel_redu_factor
        )

    if out_op == 'conv':
      nn.Conv2d(self.num_curvelets, 1, kernel_size=1)
    elif out_op == 'sum':
      self.out_synthesizer = Reduce('b c h w -> b 1 h w', 'sum')
    else:
      raise ValueError('out op can only be `conv`, or `sum`')

  def freeze_unet(self):
    """Freeze the UNet"""
    for param in self.unet.parameters():
      param.requires_grad = False

  def unfreeze_unet(self):
    """Freeze the UNet"""
    for param in self.unet.parameters():
      param.requires_grad = True

  def freeze_router(self):
    """Unfreeze the Router"""
    for param in self.router_1.parameters():
      param.requires_grad = False
    if self.use_two_routers:
      for param in self.router_2.parameters():
        param.requires_grad = False

  def unfreeze_router(self):
    """Unfreeze the Router"""
    for param in self.router_1.parameters():
      param.requires_grad = True
    if self.use_two_routers:
      for param in self.router_2.parameters():
        param.requires_grad = True

  def route_bands(self, x_bands):
    """Route the bands of the input image
    Args:
        x_bands (torch.FloatTensor): the bands of the input image
            that has the size [batchsize, num_curvelets, sidelen, sidelen]

    Returns:
        warped_bands (torch.FloatTensor): the warped bands of the
            input image that has the size
            [batchsize, num_curvelets, sidelen, sidelen]
    """
    warped_bands = self.router_1.warp(self.router_input, x_bands)

    if self.use_two_routers:
      warped_bands = warped_bands + self.router_2.warp(
          self.router_input, warped_bands)

    return warped_bands

  def forward(self, x, router_only: bool):
    # pylint: disable=invalid-name
    """Forward pass of the FioNet
    Args:
        x (torch.FloatTensor): FioNet input that has
            the size [batchsize, 1, sidelen, sidelen].

    Returns:
        out (torch.FloatTensor): FioNet output
    """
    x_bands = self.decomposer(x)
    # [batchsize, num_curvelets, sidelen, sidelen].

    if router_only:
      warped_bands = self.route_bands(x_bands)
    else:
      x_bands = self.unet(x_bands)
      warped_bands = self.route_bands(x_bands)

    out = self.out_synthesizer(warped_bands)
    out = self.out_activation(out)
    # [batchsize, 1, sidelen, sidelen]
    return out
