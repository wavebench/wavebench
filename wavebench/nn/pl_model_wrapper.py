"""Pytorch lightning wrapper for models"""""
import torch
import pytorch_lightning as pl
from wavebench.nn.unet import UNet
from wavebench.nn.fno import FNO2d
from wavebench.nn.lploss import LpLoss

def get_model(model_config):
  model_config = model_config.copy()
  model_name = model_config.pop('model_name')
  if model_name.lower() == 'unet':
    model = UNet(**model_config)
  elif model_name.lower() == 'fno':
    model = FNO2d(**model_config)
  else:
    raise ValueError('Unknown model name: {}'.format(model_name))
  return model


class LitModel(pl.LightningModule):
  """ Pytorch lightning wrapper for models"""""
  def __init__(
    self,
    model_config,
    max_num_steps: int,
    eta_min: float,
    weight_decay: float = 0.0,
    learning_rate: float = 1e-3,
    loss_fun_type: str = 'mse'):
    super().__init__()

    self.model = get_model(model_config)
    self.max_num_steps = max_num_steps
    self.eta_min = eta_min
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.save_hyperparameters()

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

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    x, y = batch

    output = self(x)
    train_loss = self.loss_fun(output, y)
    self.log('train_loss', train_loss.detach(), prog_bar=True)
    return train_loss

  def validation_step(self, batch, batch_idx):
    # pylint:disable=missing-function-docstring
    # pylint:disable=invalid-name
    x, y = batch
    output = self(x)
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
