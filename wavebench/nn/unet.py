""" U-Net model modified from https://github.com/milesial/Pytorch-UNet"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
  """(convolution => [BN] => ReLU) * 2"""

  def __init__(self, in_channels, out_channels, mid_channels=None):
    super().__init__()
    if not mid_channels:
      mid_channels = out_channels
    self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1,
                  bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)


class Down(nn.Module):
  """Downscaling with maxpool then double conv"""

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.maxpool_conv = nn.Sequential(
        nn.MaxPool2d(2),
        DoubleConv(in_channels, out_channels)
    )

  def forward(self, x):
    return self.maxpool_conv(x)


class Up(nn.Module):
  """Upscaling then double conv"""

  def __init__(self, in_channels, out_channels, bilinear=True):
    super().__init__()

    # if bilinear, use the normal convolutions to reduce the
    # number of channels
    if bilinear:
      self.up = nn.Upsample(
        scale_factor=2, mode='bilinear',
        align_corners=True)
      self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    else:
      self.up = nn.ConvTranspose2d(
        in_channels,
        in_channels // 2,
        kernel_size=2, stride=2)
      self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
    x1 = self.up(x1)
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)


class OutConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(OutConv, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

  def forward(self, x):
    return self.conv(x)


class UNet(nn.Module):
  def __init__(self, n_input_channels, n_output_channels,
              channel_reduction_factor=1,
              bilinear=False):
    super(UNet, self).__init__()
    self.n_input_channels = n_input_channels
    self.n_output_channels = n_output_channels
    self.bilinear = bilinear

    channels_nums = [64, 128, 256, 512, 1024]
    channels_nums = [channel // channel_reduction_factor
                      for channel in channels_nums]
    self.channels_nums = channels_nums
    self.inc = DoubleConv(n_input_channels, channels_nums[0])
    self.down1 = Down(channels_nums[0], channels_nums[1])
    self.down2 = Down(channels_nums[1], channels_nums[2])
    self.down3 = Down(channels_nums[2], channels_nums[3])
    factor = 2 if bilinear else 1
    self.down4 = Down(channels_nums[3], channels_nums[4] // factor)
    self.up1 = Up(channels_nums[4], channels_nums[3] // factor, bilinear)
    self.up2 = Up(channels_nums[3], channels_nums[2] // factor, bilinear)
    self.up3 = Up(channels_nums[2], channels_nums[1] // factor, bilinear)
    self.up4 = Up(channels_nums[1], channels_nums[0], bilinear)
    self.outc = OutConv(channels_nums[0], n_output_channels)

  def forward(self, x):
    x1 = self.inc(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)
    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)
    logits = self.outc(x)
    return logits
