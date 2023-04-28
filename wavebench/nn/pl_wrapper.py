"""Pytorch lightning wrapper for models"""""
import torch
import pytorch_lightning as pl
from fionet.unet import UNet
from neuralop.models import FNO2d

def get_model(model_config):
  model_name = model_config.pop('name')
  if model_name == 'UNet':
    model = UNet(**model_config)
  elif model_name == 'FNO':
    model = FNO2d(**model_config)
  return model


class LitModel(pl.LightningModule):
  def __init__(self,
              model_config,
              max_num_steps: int,
              eta_min: float,
              weight_decay: float = 0.0,
              learning_rate: float = 1e-4):

    super().__init__()

    self.model = get_model(model_config)
    self.max_num_steps = max_num_steps
    self.eta_min = eta_min
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.save_hyperparameters()
    self.loss_fun = torch.nn.MSELoss()

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    x, y = batch

    output = self(x)
    mse_loss = self.loss_fun(output, y)
    self.log('train_mse_loss', mse_loss.detach(), prog_bar=True)
    return mse_loss

  def validation_epoch_end(self, outputs):
    val_loss = torch.stack(
        [x['val_mse_loss'] for x in outputs]).mean()
    self.log('val_mse_loss', val_loss)

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.parameters(),
        lr=self.learning_rate,
        weight_decay=self.weight_decay,
        eps=self.eps)

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

