""" Pytorch Lightning wrapper for FIONet. """
import torch
import pytorch_lightning as pl
from wavebench.nn.fionet_module.fionet import FIONet
from wavebench.nn.lploss import LpLoss


class LitFIONet(pl.LightningModule):
  """ Pytorch Lightning module of FIONet. """
  def __init__(self,
              max_num_steps: int,
              eta_min: float,
              use_two_routers: bool = False,
              router_use_curvelet_scales=False,
              keep_only_curvelet_middle_scales=True,
              unet_channel_redu_factor=2,
              sidelen=128,
              n_output_channels=1,
              siren_latent_dim=512,
              siren_num_layers=5,
              siren_omega=30.,
              siren_c=6.,
              out_op='sum',
              out_activation='id',
              weight_decay: float = 0.0,
              learning_rate: float = 1e-3,
              loss_fun_type: str = 'mse',
              ):

    super().__init__()
    self.save_hyperparameters()

    model_kwargs = {
        'sidelen': sidelen,
        'n_output_channels': n_output_channels,
        'use_two_routers': use_two_routers,
        'router_use_curvelet_scales':
            router_use_curvelet_scales,
        'keep_only_curvelet_middle_scales':
            keep_only_curvelet_middle_scales,
        'siren_latent_dim': siren_latent_dim,
        'siren_num_layers': siren_num_layers,
        'siren_omega': siren_omega,
        'siren_c': siren_c,
        'out_op': out_op,
        'out_activation': out_activation,
        'unet_channel_redu_factor': unet_channel_redu_factor
        }

    self.model = FIONet(**model_kwargs)
    self.model.freeze_unet()

    self.max_num_steps = max_num_steps
    self.eta_min = eta_min
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    # self.loss_fun = torch.nn.MSELoss()

    self.mse_loss = torch.nn.MSELoss()
    self.l1_loss = torch.nn.L1Loss()
    self.lp_loss = LpLoss(p=2)

    if loss_fun_type == 'mse':
      self.loss_fun = self.mse_loss
    elif loss_fun_type == 'l1':
      self.loss_fun = self.l1_loss
    elif loss_fun_type == 'relative_l2':
      self.loss_fun = self.lp_loss
    else:
      raise ValueError(
        'Unknown loss function type: {}'.format(loss_fun_type))


    # print(f'trainer.max_steps: {self.trainer.max_steps}')
    # print(f'max_num_steps: {max_num_steps}')

    # assert self.trainer.max_steps == max_num_steps

  def forward(self, x, router_only):  # pylint: disable=arguments-differ
    """Forward pass"""
    return self.model(x, router_only)

  def training_step(self, batch, batch_idx):
    # pylint: disable=arguments-differ, unused-argument, invalid-name
    x, y = batch

    if self.global_step == self.max_num_steps // 2:
      self.model.unfreeze_unet()
      self.model.freeze_router()
      print(f'Unfreezing UNet & Freezing router at step {self.global_step}.')

    router_only = self.global_step < self.max_num_steps // 2
    output = self.model(
      x,
      router_only=router_only)

    train_loss = self.loss_fun(output, y)
    self.log(
      'train_loss', train_loss.detach(),
        prog_bar=True,
        sync_dist=True)
    self.log(
      'router_only', router_only,
        prog_bar=True,
        sync_dist=True)
    return train_loss

  def validation_step(self, batch, batch_idx):
    # pylint: disable=arguments-differ, unused-argument, invalid-name
    x, y = batch
    output = self.model(
      x,
      router_only=self.trainer.global_step < self.max_num_steps // 2)

    val_mse_loss = self.mse_loss(output, y).detach()
    self.log("val_mse_loss",
             val_mse_loss,
             on_epoch=True,
             prog_bar=True,
             sync_dist=True)

    val_rel_lp_loss = self.lp_loss(output, y).detach()
    self.log("val_rel_lp_loss",
             val_rel_lp_loss,
             on_epoch=True,
             prog_bar=True,
             sync_dist=True)

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.parameters(),
        lr=self.learning_rate,
        weight_decay=self.weight_decay
        )

    sched_config = {
        'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=self.max_num_steps,
        eta_min=self.eta_min),
        'interval': 'step',
        'frequency': 1
    }
    # https://github.com/Lightning-AI/lightning/issues/9475
    return {"optimizer": optimizer, "lr_scheduler": sched_config}
