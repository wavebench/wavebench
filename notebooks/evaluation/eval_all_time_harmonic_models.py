""" Evaluate all models on the time-harmonic datasets"""
import pprint
from pathlib import Path
import wandb
import matplotlib.pyplot as plt
import ml_collections
from pytorch_lightning.loggers import WandbLogger

from wavebench.dataloaders.helmholtz_loader import get_dataloaders_helmholtz


from wavebench import wavebench_figure_path
from wavebench.nn.pl_model_wrapper import LitModel
from wavebench import wavebench_checkpoint_path
from wavebench.plot_utils import plot_images, remove_frame



all_models = [
  {
    "tag": 'fno-depth-4',
    "config.model_config/model_name": 'fno',
    "config.model_config/num_hidden_layers": 4
  },
  {
    "tag": 'fno-depth-8',
    "config.model_config/model_name": 'fno',
    "config.model_config/num_hidden_layers": 8
  },
  {
    "tag": 'unet-ch-32',
    "config.model_config/model_name": 'unet',
    "config.model_config/channel_reduction_factor": 2
  },
  {
    "tag": 'unet-ch-64',
    "config.model_config/model_name": 'unet',
    "config.model_config/channel_reduction_factor": 1
  },

              ]


# Initialize the W&B API client
api = wandb.Api()


pp = pprint.PrettyPrinter(depth=6)
device = 'cpu'

eval_config = ml_collections.ConfigDict()


# problem setting: can be 'isotropic' or 'anisotropic'
# for eval_config.kernel_type in ['isotropic', 'anisotropic']:
for eval_config.kernel_type in ['isotropic']:

  # frequency: can be in [1.0, 1.5, 2.0, 4.0]
  # for eval_config.frequency in [1.0, 1.5, 2.0, 4.0]:
  for eval_config.frequency in [1.0]:

    test_loader = get_dataloaders_helmholtz(
      eval_config.kernel_type,
      eval_config.frequency)['test']
    model_dict = {}

    for model_filters in all_models:
      _model_filters = model_filters.copy()

      model_tag = _model_filters.pop('tag')

      project = f'helmholtz_{eval_config.kernel_type}_{eval_config.frequency}'
      runs = api.runs(
        path=f"tliu/{project}",
        filters=_model_filters)

      # make sure that there is a unique model that satisfies the filters
      assert len(runs) == 1

      run_id = runs[0].id

      checkpoint_reference = f"tliu/{project}/model-{run_id}:best"
      print(f'checkpoint: {checkpoint_reference}')

      # delete all the checkpoints that do not have the aliases such as 'best
      artifact_versions = api.artifact_versions(
        name=f'{project}/model-{run_id}', type_name='model')

      for v in artifact_versions:
        if len(v.aliases) == 0:
          v.delete()
          print(f'deleted {v.name}')
        else:
          print(f'kept {v.name}, {v.aliases}')

      artifact_dir = WandbLogger.download_artifact(
        artifact=checkpoint_reference,
        save_dir=wavebench_checkpoint_path)

      # load checkpoint
      model = LitModel.load_from_checkpoint(
        Path(artifact_dir) / "model.ckpt").to(device)

      print('model hparams:')
      pp.pprint(model.hparams.model_config)

      model_dict[model_tag] = model

      sample_input, sample_target = next(iter(test_loader))

      pred_dict_real = {
        'input': sample_input.squeeze(),
        'gt': sample_target.squeeze()[0]}

      # pred_dict_img = {
      #   'input': sample_input.squeeze(),
      #   'gt': sample_target.squeeze()[1]}

      print(len(list(pred_dict_real.keys())))

      for i in range(len(model_dict.items()) - 2):
        # placeholder for the plots; these plots will be removed later
        pred_dict_real[f'placeholder_{i}'] = sample_target.squeeze()[0]
        # pred_dict_img[f'placeholder_{i}'] = sample_target.squeeze()[1]

        print(len(list(pred_dict_real.keys())))

      for tag, model in model_dict.items():
        pred = model(
          sample_input.to(device)).detach().cpu().squeeze()
        pred_dict_real[tag] = pred.squeeze()[0]
        # pred_dict_img[tag] = pred.squeeze()[1]

        pred_dict_real[f'{tag}_diff'] = (
          pred.squeeze()[0] - sample_target.squeeze()[0]).abs()
        # pred_dict_img[f'{tag}_diff'] = (
        #   pred.squeeze()[1] - sample_target.squeeze()[1]).abs()
        print(len(list(pred_dict_real.keys())))

      fig, axes = plot_images(
        list(pred_dict_real.values()),
        nrows=3, # ground-truth row, real prediction row, real difference row
        cbar='one',
        vrange='individual',
        fig_size=(9, 3),
        shrink=0.5,
        pad=0.02,
        cmap='coolwarm')

      # axes[0,0]
      for i, ax in enumerate(axes.flatten()):
        if list(pred_dict_real.keys())[i].startswith('placeholder'):
          ax.remove()
        else:
          ax.set_title( list(pred_dict_real.keys()) [i])
          remove_frame(ax)

      plt.suptitle(
        f'Real part. Wavespeed: {eval_config.kernel_type}, Freq: {eval_config.frequency}',
        y=0.62,)

      plt.savefig(
        f"{wavebench_figure_path}/model_out_real_{eval_config.kernel_type}_{eval_config.frequency}.pdf",
        format="pdf", bbox_inches="tight")

      # imaginery part
      # fig, axes = plot_images(
      #   list(pred_dict_img.values()),
      #   cbar='none',
      #   vrange='individual',
      #   fig_size=(9, 9),
      #   shrink=0.15,
      #   pad=0.02,
      #   cmap='coolwarm')

      # # axes[0,0]
      # for i, ax in enumerate(axes.flatten()):
      #   ax.set_title( list(pred_dict_img.keys()) [i])
      #   remove_frame(ax)

      # plt.suptitle(
      #   f'Img part. Wavespeed: {eval_config.kernel_type}, Freq: {eval_config.frequency}',
      #   y=0.62,)

      # plt.savefig(
      #   f"{wavebench_figure_path}/model_out_img_{eval_config.kernel_type}_{eval_config.frequency}.pdf",
      #   format="pdf", bbox_inches="tight")

