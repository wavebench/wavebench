""" Train the models on time-harmonic wavebench datasets. """
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger #TensorBoardLogger

from wavebench import wavebench_path
from wavebench.nn.pl_model_wrapper import LitModel
from wavebench.dataloaders.helmholtz_loader import get_dataloaders_helmholtz


parser = argparse.ArgumentParser()

# Dataset settings
parser.add_argument('--batch_size', type=int, default=32,
    help='The mini-batch size for training.')
parser.add_argument('--kernel_type', type=str, default='isotropic',
    help='Can be `isotropic` or `anisotropic`.')
parser.add_argument('--frequency', type=float, default=1.0,
    help='Can be 1.0, 1.5, 2.0, 4.0 ')


# Model settings
parser.add_argument('--channel_reduction_factor', type=int, default=2,
    help='Channel redu factor.')

# Training settings
parser.add_argument('--num_epochs', type=int, default=20,
                    help='number of training epochs.')
parser.add_argument('--loss_fun_type', type=str, default='relative_l2',
                    help='the loss function.')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='learning rate of gradient descent.')
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--eta_min', type=float, default=1e-5,
                    help='the eta_min for CosineAnnealingLR decay.')


# DistributedDataParallel settings
parser.add_argument('--num_workers', type=int, default=20,
                    help='')
parser.add_argument('--gpu_devices', type=int, nargs='+', default=[1],
                    help='')


def main():
  """Main function."""
  pl.seed_everything(42)
  args = parser.parse_args()


  dataset_setting_dict = {
      'train_batch_size': args.batch_size,
      'eval_batch_size': args.batch_size,
      'num_workers': args.num_workers,
      'kernel_type': args.kernel_type,
      'frequency': args.frequency,
      }

  loaders = get_dataloaders_helmholtz(
      **dataset_setting_dict)


  model_config = {
    'model_name': 'unet',
    'n_input_channels': 1,
    'n_output_channels': 2,
    'channel_reduction_factor': args.channel_reduction_factor
    }

  model_name = model_config['model_name']

  training_config = {
      'max_num_steps': args.num_epochs * len(
        loaders['train']) // len(args.gpu_devices),
      'eta_min': args.eta_min,
      'weight_decay': args.weight_decay,
      'learning_rate': args.learning_rate,
      'loss_fun_type': args.loss_fun_type,
      }

  model = LitModel(
      model_config=model_config,
      **training_config)

  checkpoint_callback = ModelCheckpoint(
      monitor='val_rel_lp_loss',
      save_top_k=1,
      mode='min')

  task_name = f'helmholtz_{args.kernel_type}_{args.frequency}'

  model_save_dir = str(wavebench_path + f'/saved_models/{task_name}')

  logger = WandbLogger(
    name=f'{model_name}_redu_factor_{args.channel_reduction_factor}',
    save_dir=wavebench_path + '/saved_models/',
    project=task_name,
    log_model="all"
    )

  logger.log_hyperparams(model.hparams)

  lr_monitor = LearningRateMonitor(logging_interval='step')
  trainer = pl.Trainer(
      profiler='simple',
      logger=logger,
      devices=args.gpu_devices,
      accelerator='gpu',
      max_epochs=args.num_epochs,
      callbacks=[checkpoint_callback, lr_monitor],
      default_root_dir=model_save_dir + '/' + task_name,
      )

  trainer.fit(
      model,
      train_dataloaders=loaders['train'],
      val_dataloaders=loaders['test'])


if __name__ == '__main__':
  main()
