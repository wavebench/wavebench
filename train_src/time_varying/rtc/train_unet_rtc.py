""" Train the models on time-varying wavebench datasets. """
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from wavebench import wavebench_path
from wavebench.nn.pl_model_wrapper import LitModel
from wavebench.dataloaders.rtc_loader import get_dataloaders_rtc_thick_lines


parser = argparse.ArgumentParser(description='U-Net training')

# Dataset settings
parser.add_argument('--batch_size', type=int, default=64,
    help='The mini-batch size for training.')
parser.add_argument('--medium_type', type=str, default='gaussian_lens',
    help='Can be `gaussian_lens` or `gaussian_random_field`.')


# Training settings
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of training epochs.')
parser.add_argument('--loss_fun_type', type=str, default='relative_l2',
                    help='the loss function.')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='learning rate of gradient descent.')
parser.add_argument('--weight_decay', type=float, default=1e-5)
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
    'num_workers': args.num_workers
    }

  loaders = get_dataloaders_rtc_thick_lines(
    medium_type=args.medium_type, **dataset_setting_dict)


  model_config = {
    'model_name': 'unet',
    'n_input_channels': 1,
    'n_output_channels': 1}

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

  task_name = f'rtc_{args.medium_type}'


  logger = WandbLogger(
    name=model_name,
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
    # default_root_dir=model_save_dir + '/' + task_name,
    )

  trainer.fit(model,
              train_dataloaders=loaders['train'],
              val_dataloaders=loaders['val'])


if __name__ == '__main__':
  main()
