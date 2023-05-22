""" Evaluate all models on the time-varying datasets"""
import pprint
import os
from pathlib import Path
import wandb
import matplotlib.pyplot as plt
import ml_collections
from pytorch_lightning.loggers import WandbLogger
from wavebench.dataloaders.rtc_loader import get_dataloaders_rtc_thick_lines, get_dataloaders_rtc_mnist
from wavebench.dataloaders.is_loader import get_dataloaders_is_thick_lines, get_dataloaders_is_mnist
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
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

eval_config = ml_collections.ConfigDict()

save_path = f'{wavebench_figure_path}/time_varying'
if not os.path.exists(save_path):
  os.makedirs(save_path)


# problem setting: can be 'is' or 'rtc'
for eval_config.problem in ['is', 'rtc']:
# for eval_config.problem in ['is']:

  # medium_type setting: can be 'gaussian_lens' or 'grf_isotropic' or 'grf_anisotropic'
  for eval_config.medium_type in ['gaussian_lens', 'grf_isotropic', 'grf_anisotropic']:

    if eval_config.problem == 'rtc':
      in_dist_test_loader = get_dataloaders_rtc_thick_lines(
        medium_type=eval_config.medium_type,
        use_ffcv=False,
      )['test']
      ood_test_loader = get_dataloaders_rtc_mnist(
        medium_type=eval_config.medium_type,
        use_ffcv=False,
      )

    elif eval_config.problem == 'is':
      in_dist_test_loader = get_dataloaders_is_thick_lines(
        medium_type=eval_config.medium_type,
        use_ffcv=False,
      )['test']
      ood_test_loader = get_dataloaders_is_mnist(
        medium_type=eval_config.medium_type,
        use_ffcv=False,
        )
    else:
      raise ValueError(
        'Cannot find the dataset with the given settings:' +
        f'problem: {eval_config.problem}, medium_type: {eval_config.medium_type}'
                        )
    model_dict = {}

    for model_filters in all_models:
      _model_filters = model_filters.copy()

      model_tag = _model_filters.pop('tag')

      project = f'{eval_config.problem}_{eval_config.medium_type}'
      runs = api.runs(
        path=f"tliu/{project}",
        filters=_model_filters)

      # make sure that there is a unique model that satisfies the filters
      assert len(runs) == 1

      run_id = runs[0].id

      checkpoint_reference = f"tliu/{project}/model-{run_id}:best"
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
      # runs = api.runs(path="tliu/rtc_gaussian_lens")

    idx = 1
    in_dist_sample_input, in_dist_sample_target = in_dist_test_loader.dataset.__getitem__(idx)
    ood_sample_input, ood_sample_target = ood_test_loader.dataset.__getitem__(idx)

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
        pred = model(sample_input.unsqueeze(0).to(device)).detach().cpu().squeeze()
        pannel_dict[f'{setting}_{tag}'] = pred

    fig, axes = plot_images(
      list(pannel_dict.values()),
      nrows=3,
      cbar='one',
      fig_size=(10, 6),
      shrink=0.45,
      pad=0.02,
      cmap='coolwarm')

    # axes[0,0]
    for i, ax in enumerate(axes.flatten()):
      ax.set_title( list(pannel_dict.keys()) [i])
      remove_frame(ax)

    plt.suptitle(
      f'Problem: {eval_config.problem}, Wavespeed: {eval_config.medium_type}',
      y=1.0,)

    plt.savefig(
      f"{save_path}/{eval_config.problem}_{eval_config.medium_type}.pdf",
      format="pdf", bbox_inches="tight")


    # idx = 1
    # sample_input, sample_target = test_loader.dataset.__getitem__(idx)

    # fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # axes[0].imshow(sample_input.squeeze().numpy(), cmap='coolwarm')
    # axes[0].set_title('Input')
    # axes[1].imshow(sample_target.squeeze().numpy(), cmap='coolwarm')
    # axes[1].set_title('Target')


    # pred_dict = {
    #   'input': sample_input.squeeze(),
    #   'gt': sample_target.squeeze()}

    # for i in range(len(model_dict.items()) - 2):
    #   # placeholder for the plots; these plots will be removed later
    #   pred_dict[f'placeholder_{i}'] = sample_target.squeeze()


    # for tag, model in model_dict.items():
    #   pred = model(
    #     sample_input.unsqueeze(0).to(device)).detach().cpu().squeeze()
    #   pred_dict[tag] = pred

    # fig, axes = plot_images(
    #   list(pred_dict.values()),
    #   nrows=2,
    #   cbar='one',
    #   fig_size=(9, 4),
    #   shrink=0.5,
    #   pad=0.02,
    #   cmap='coolwarm')

    # # axes[0,0]
    # for i, ax in enumerate(axes.flatten()):
    #   if list(pred_dict.keys())[i].startswith('placeholder'):
    #     ax.remove()
    #   else:
    #     ax.set_title( list(pred_dict.keys()) [i])
    #     remove_frame(ax)

    # plt.suptitle(
    #   f'Problem: {eval_config.problem}, Init pressure: {eval_config.dataset_name}, Wavespeed: {eval_config.medium_type}',
    #   y=1.0,)

