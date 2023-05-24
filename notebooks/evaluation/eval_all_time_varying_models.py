""" Evaluate all models on the time-varying datasets"""

#%%
import pprint
import os
from pathlib import Path
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import ml_collections

from pytorch_lightning.loggers import WandbLogger
from wavebench.dataloaders.rtc_loader import get_dataloaders_rtc_thick_lines, get_dataloaders_rtc_mnist
from wavebench.dataloaders.is_loader import get_dataloaders_is_thick_lines, get_dataloaders_is_mnist
from wavebench import wavebench_figure_path
from wavebench.nn.pl_model_wrapper import LitModel
from wavebench import wavebench_checkpoint_path
from wavebench.plot_utils import plot_image, remove_ticks
from wavebench.nn.lploss import lp_loss



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
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cuda:0'

eval_config = ml_collections.ConfigDict()
eval_config.wandb_entity = 'tliu'

save_path = f'{wavebench_figure_path}/time_varying'
if not os.path.exists(save_path):
  os.makedirs(save_path)


in_dist_test_performance_dict = {}
ood_test_performance_dict = {}

def evaluate_model_on_loader(model, loader, device=device):
  model.eval()
  model.to(device)
  total_loss = 0
  with torch.no_grad():
    for sample_input, sample_target in loader:
      pred = model(sample_input.to(device)).detach().cpu()
      loss = lp_loss(pred, sample_target).item()
      total_loss += loss

  return total_loss/len(loader)

# problem setting: can be 'is' or 'rtc'
for eval_config.problem in [
  'rtc',
  'is'
  ]:
# for eval_config.problem in ['is']:

  # medium_type setting: can be 'gaussian_lens'
  # or 'grf_isotropic' or 'grf_anisotropic'
  for eval_config.medium_type in [
    'gaussian_lens',
    'grf_isotropic',
    'grf_anisotropic'
    ]:

    if eval_config.problem == 'rtc':
      in_dist_test_loader = get_dataloaders_rtc_thick_lines(
        medium_type=eval_config.medium_type,
      )['test']
      ood_test_loader = get_dataloaders_rtc_mnist(
        medium_type=eval_config.medium_type,
      )

    elif eval_config.problem == 'is':
      in_dist_test_loader = get_dataloaders_is_thick_lines(
        medium_type=eval_config.medium_type,
      )['test']
      ood_test_loader = get_dataloaders_is_mnist(
        medium_type=eval_config.medium_type,
        )
    else:
      raise ValueError(
        'Cannot find the dataset with the given settings:' +\
        f'problem: {eval_config.problem}' +\
        f'medium_type: {eval_config.medium_type}'
                        )

    in_dist_test_performance_dict[
      f'{eval_config.problem}_{eval_config.medium_type}'] = []

    ood_test_performance_dict[
      f'{eval_config.problem}_{eval_config.medium_type}'] = []


    model_dict = {}

    for model_filters in all_models:
      _model_filters = model_filters.copy()

      model_tag = _model_filters.pop('tag')

      project = f'{eval_config.problem}_{eval_config.medium_type}'
      runs = api.runs(
        path=f"{eval_config.wandb_entity}/{project}",
        filters=_model_filters)

      # make sure that there is a unique model that satisfies the filters
      assert len(runs) == 1

      run_id = runs[0].id

      checkpoint_reference = f"{eval_config.wandb_entity}/{project}" +\
        f"/model-{run_id}:best"
      print(f'checkpoint: {checkpoint_reference}')

      # delete all the checkpoints that do not have aliases such as
      # 'best' or 'latest'.
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

      in_dist_test_performance_dict[
        f'{eval_config.problem}_{eval_config.medium_type}'].append(
          evaluate_model_on_loader(model, in_dist_test_loader))

      ood_test_performance_dict[
        f'{eval_config.problem}_{eval_config.medium_type}'].append(
          evaluate_model_on_loader(model, ood_test_loader))


    in_dist_sample_input, in_dist_sample_target = next(iter(
      in_dist_test_loader))
    ood_sample_input, ood_sample_target = next(iter(
      ood_test_loader))

    pannel_dict = {
      'in-distri input': in_dist_sample_input.squeeze(),
      'in-distri target': in_dist_sample_target.squeeze(),
      'out-of-distri input': ood_sample_input.squeeze(),
      'out-of-distri target': ood_sample_target.squeeze(),
    }

    for setting in ['in_dist', 'ood']:
      if setting == 'in_dist':
        sample_input = in_dist_sample_input
      else:
        sample_input = ood_sample_input
      for tag, model in model_dict.items():
        model.eval()
        model.to(device)
        pred = model(
          sample_input.to(device)).detach().cpu().squeeze()
          # sample_input.unsqueeze(0).to(device)).detach().cpu().squeeze()
        pannel_dict[f'{setting}_{tag}'] = pred

    nrows = 3
    ncols = 4
    fig_size = (10, 7.5)
    cbar_shrink = 0.7
    x_list = list(pannel_dict.values())

    for i, x in enumerate(x_list):
      x_list[i] = np.asarray(x)

    fig = plt.figure()
    fig.set_size_inches(fig_size)
    axes = fig.subplots(nrows, ncols)
    im = np.empty(axes.shape, dtype=object)

    for i, (x, ax_) in enumerate(zip(x_list, axes.flat)):
      if i ==0 or i == 2:
        im_, _ = plot_image(
          x, ax=ax_, norm=colors.CenteredNorm(), cmap='seismic')
      else:
        im_, _ = plot_image(x, ax=ax_, vmin=0., vmax=1., cmap='jet')
      im.flat[i] = im_
      fig.colorbar(im_, ax=ax_, shrink=cbar_shrink)

    for i, ax in enumerate(axes.flatten()):
      ax.set_title( list(pannel_dict.keys()) [i])
      remove_ticks(ax)

    plt.suptitle(
      f'Problem: {eval_config.problem}, Wavespeed: {eval_config.medium_type}',
      y=1.0,)

    plt.savefig(
      f"{save_path}/{eval_config.problem}_{eval_config.medium_type}.pdf",
      format="pdf", bbox_inches="tight")


#%%

import pandas as pd

columns = [a['tag'] for a in all_models]
in_dist_df = pd.DataFrame.from_dict(
  in_dist_test_performance_dict, orient='index', columns=columns)

# round the values
for column in columns:
  in_dist_df[column] = in_dist_df[column].round(3)
in_dist_df

# %%
ood_df = pd.DataFrame.from_dict(
  ood_test_performance_dict, orient='index', columns=columns)

# round the values
for column in columns:
  ood_df[column] = ood_df[column].round(3)
ood_df
