"""UNO model. """
import torch.nn.functional as F
import torch.nn as nn
import torch
from wavebench.nn.fno import Conv1x1, FourierLayer2d, concat_coordinates

class UNO2d(nn.Module):
  """A UNO2d model. """
  def __init__(self,
               num_in_channels=1,
               num_out_channels=1,
               lifting_channels=128,
               projection_channels=128,
               base_hidden_width=32,
               modes1=12,
               modes2=12
               ):
    super().__init__()

    self.num_in_channels = num_in_channels
    self.num_out_channels = num_out_channels
    self.lift_channels = lifting_channels
    self.proj_channels = projection_channels
    self.width = base_hidden_width



    self.lifter = nn.Sequential(
      Conv1x1(num_in_channels+2,
              lifting_channels,
              ),
      nn.GELU(),
      Conv1x1(lifting_channels,
              base_hidden_width,
              ),
    )

    self.projector = nn.Sequential(
      Conv1x1(base_hidden_width,
              projection_channels),
      nn.GELU(),
      Conv1x1(projection_channels,
              num_out_channels),
    )

    # pad the domain if input is non-periodic
    self.padding = 9

    channels_nums = [base_hidden_width, base_hidden_width * 2, base_hidden_width * 4]

    self.inc = FourierLayer2d(
      num_in_channels=channels_nums[0],
      num_out_channels=channels_nums[0],
      modes1=modes1,
      modes2=modes2,
    )

    self.down1 = FourierLayer2d(
      num_in_channels=channels_nums[0],
      num_out_channels=channels_nums[1],
      modes1=modes1,
      modes2=modes2,
    )

    self.down2 = FourierLayer2d(
      num_in_channels=channels_nums[1],
      num_out_channels=channels_nums[2],
      modes1=modes1//2,
      modes2=modes2//2,
    )

    self.bottoneck = FourierLayer2d(
      num_in_channels=channels_nums[2],
      num_out_channels=channels_nums[2],
      modes1=modes1//2,
      modes2=modes2//2,
      )

    self.up1 = FourierLayer2d(
      num_in_channels=channels_nums[2],
      num_out_channels=channels_nums[1],
      modes1=modes1//2,
      modes2=modes2//2,
    )
    self.up2 = FourierLayer2d(
      num_in_channels=2*channels_nums[1],
      num_out_channels=channels_nums[0],
      modes1=modes1,
      modes2=modes2,
    )


    self.out = FourierLayer2d(
      num_in_channels=2*channels_nums[0],
      num_out_channels=channels_nums[0],
      modes1=modes1,
      modes2=modes2,
      )


  def forward(self, x):
    """Forward pass. """

    # lifting
    x = concat_coordinates(x)
    x = self.lifter(x)

    # encoder
    x = F.pad(x, [0,self.padding, 0,self.padding])
    h, w = x.shape[-2], x.shape[-1]
    x0 = self.inc(x) # [B, c, h, w]
    x1 = self.down1(x0, h//2, w//2) # [B, 2*c, h//2, w//2]
    x2 = self.down2(x1, h//4, w//4) # [B, 4*c, h//4, w//4]
    x2 = self.bottoneck(x2, h//4, w//4) # [B, 8*c, h//4, w//4]

    # decoder
    x = self.up1(x2, h//2, w//2) # [B, 2*c, h//2, w//2]
    x = self.up2(self.concat(x, x1), h, w) # [B, c, h, w]
    x = self.out(self.concat(x, x0), h, w) # [B, c, h, w]
    x = x[..., :-self.padding, :-self.padding]

    # projection
    x = self.projector(x)
    return x

  def concat(self, x1, x2):
    x = torch.cat([x2, x1], dim=1)
    return x
