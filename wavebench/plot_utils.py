"""Common utils for used by different dataset builders.

Many of these were taken from the dival library:
https://github.com/jleuschn/dival

Signal propagation plots (SPP) for PyTorch models are taken from:
https://github.com/mehdidc/signal_propagation_plot/

"""

from math import ceil
from functools import partial
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np


def simpleaxis(ax):
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.get_xaxis().tick_bottom()
  ax.get_yaxis().tick_left()


def remove_frame(ax):
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.get_xaxis().set_ticks([])
  ax.get_yaxis().set_ticks([])

SPINE_COLOR = 'gray'
def remove_ticks(ax):
  for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_color(SPINE_COLOR)
    ax.spines[spine].set_linewidth(0.5)
  ax.get_xaxis().set_ticks([])
  ax.get_yaxis().set_ticks([])

def plot_image(x, fig=None, ax=None, **kwargs):
  """Plot image using matplotlib's :meth:`imshow` method.

  Parameters
  ----------
  x : array-like or PIL image
      The image data. For further information see `imshow documentation
      <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html>`_.
  fig : :class:`matplotlib.figure.Figure`, optional
      The figure to plot the image in. If ``fig is None``, but `ax` is given,
      it is retrieved from `ax`. If both ``fig is None`` and ``ax is None``,
      a new figure is created.
  ax : :class:`matplotlib.axes.Axes`, optional
      The axes to plot the image in. If `None`, an axes object is created
      in `fig`.
  kwargs : dict, optional
      Keyword arguments passed to ``ax.imshow``.

  Returns
  -------
  im : :class:`matplotlib.image.AxesImage`
      The image that was plotted.
  ax : :class:`matplotlib.axes.Axes`
      The axes the image was plotted in.
  """
  if fig is None:
    if ax is None:
      fig = plt.figure()
    else:
      fig = ax.get_figure()
  if ax is None:
    ax = fig.add_subplot(111)
  kwargs.setdefault('cmap', 'gray')
  xticks = kwargs.pop('xticks', None)
  yticks = kwargs.pop('yticks', None)
  if xticks is not None:
    ax.set_xticks(xticks)
  if yticks is not None:
    ax.set_yticks(yticks)
  im = ax.imshow(np.asarray(x), **kwargs)
  return im, ax


def plot_images(x_list, nrows=1, ncols=-1, fig=None, vrange='equal',
                cbar='auto', rect=None, fig_size=None,
                zerocenter_cbar=False, **kwargs):
  """Plot multiple images using matplotlib's :meth:`imshow` method in
  subplots.

  Parameters
  ----------
  x_list : sequence of (array-like or PIL image)
      List of the image data. For further information see `imshow
      documentation
      <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html>`_.
  nrows : int, optional
      The number of subplot rows (the default is 1). If -1, it is computed by
      ``ceil(len(x_list)/ncols)``, or set to 1 if `ncols` is not given.
  ncols : int, optional
      The number of subplot columns. If -1, it is computed by
      ``ceil(len(x_list)/nrows)`` (default). If both `nrows` and `ncols` are
      given, the value of `ncols` is ignored.
  vrange : {``'equal'``, ``'individual'``} or [list of ](float, float),\
            optional
      Value ranges for the colors of the images.
      If a string is passed, the range is auto-computed:

          ``'equal'``
              The same colors are used for all images.
          ``'individual'``
              The colors differ between the images.

      If a tuple of floats is passed, it is used for all images.
      If a list of tuples of floats is passed, each tuple is used for one
      image.
  cbar : {``'one'``, ``'many'``, ``'auto'``, ``'none'``}, optional
      Colorbar option.
      If ``cbar=='one'``, one colorbar is shown. Only possible if the value
      ranges used for the colors (cf. `vrange`) are the same for all images.
      If ``cbar=='many'``, a colorbar is shown for every image.
      If ``cbar=='auto'``, either ``'one'`` or ``'many'`` is chosen,
      depending on whether `vrange` is equal for all images.
      If ``cbar=='none'``, no colorbars are shown.
  fig : :class:`matplotlib.figure.Figure`, optional
      The figure to plot the images in. If `None`, a new figure is created.
  kwargs : dict, optional
      Keyword arguments passed to `plot_image`, which in turn passes them to
      ``imshow``.

  Returns
  -------
  im : ndarray of :class:`matplotlib.image.AxesImage`
      The images that were plotted.
  ax : ndarray of :class:`matplotlib.axes.Axes`
      The axes the images were plotted in.
  """
  cbar_shrink = kwargs.pop('shrink', 1.0)
  cbar_pad = kwargs.pop('pad', 0.04)
  try:
    x_list = list(x_list)
  except TypeError:
    raise TypeError(
        'x_list must be iterable. Pass a sequence or use '
        '`plot_image` to plot single images.')
  for i in range(len(x_list)):
    x_list[i] = np.asarray(x_list[i])
  if fig is None:
    fig = plt.figure()
  if nrows is None or nrows == -1:
    if ncols is None or ncols == -1:
      nrows = 1
    else:
      nrows = ceil(len(x_list)/ncols)
  ncols = ceil(len(x_list)/nrows)
  if rect is None:
    rect = [0.1, 0.1, 0.8, 0.8]
  if fig_size is not None:
    fig.set_size_inches(fig_size)
  if isinstance(vrange, str):
    if vrange == 'equal':
      if zerocenter_cbar:
        max_abs = max((np.max( np.abs(x)) for x in x_list))
        vrange_ = [(-max_abs, max_abs) ] * len(x_list)
      else:
        vrange_ = [(min((np.min(x) for x in x_list)),
                    max((np.max(x) for x in x_list)))] * len(x_list)
      VRANGE_EQUAL = True
    elif vrange == 'individual':
      if zerocenter_cbar:
        vrange_ = [(-np.max(np.abs(x)), np.max(np.abs(x)) ) for x in x_list]
      else:
        vrange_ = [(np.min(x), np.max(x)) for x in x_list]
      VRANGE_EQUAL = False
    else:
      raise ValueError("`vrange` must be 'equal' or 'individual'")
  elif isinstance(vrange, tuple) and len(vrange) == 2:
    vrange_ = [vrange] * len(x_list)
    VRANGE_EQUAL = True
  else:
    vrange_ = vrange
    VRANGE_EQUAL = False
  if not VRANGE_EQUAL:
    if cbar == 'one':
      warn("cannot use cbar='one' when vrange is not equal for all"
            "images, falling back to cbar='many'")
    if cbar != 'none':
      cbar = 'many'
  elif cbar == 'auto':
    cbar = 'one'
  ax = fig.subplots(nrows, ncols)
  if isinstance(ax, plt.Axes):
    ax = np.atleast_1d(ax)
  im = np.empty(ax.shape, dtype=object)
  for i, (x, ax_, v) in enumerate(zip(x_list, ax.flat, vrange_)):
    im_, _ = plot_image(x, ax=ax_, vmin=v[0], vmax=v[1], **kwargs)
    im.flat[i] = im_
    if cbar == 'many':
      # fig.colorbar(im_, ax=ax_, fraction=0.046, pad=0.04, shrink=0.5)
      fig.colorbar(im_, ax=ax_, pad=cbar_pad, shrink=cbar_shrink)

  if cbar == 'one':
    # fig.colorbar(im[0], ax=ax, fraction=0.046, pad=0.04, shrink=0.5)
    fig.colorbar(im.flatten()[0], ax=ax, pad=cbar_pad, shrink=cbar_shrink)

  return im, ax


def plot_spp(name_values, *args, **kwargs):
  labels = [name for name, value in name_values]
  values  = [value for name, value in name_values]
  depth = np.arange(len(labels))
  plt.plot(depth, values, *args, **kwargs)
  plt.xticks(depth, labels, rotation="vertical")

def get_average_channel_squared_mean_by_depth(model,  *args, **kwargs):
  acts = extract_activations(model, *args, **kwargs)
  values = []
  for name, tensor in acts:
    values.append((name, average_channel_squared_mean(tensor)))
  return values

def get_average_channel_variance_by_depth(model,  *args, **kwargs):
  acts = extract_activations(model, *args, **kwargs)
  values = []
  for name, tensor in acts:
    values.append((name, average_channel_variance(tensor)))
  return values


def average_channel_squared_mean(x):
  if x.ndim == 4:
    return (x.mean(dim=(0,2,3))**2).mean().item()
  elif x.ndim == 2:
    return (x**2).mean().item()
  else:
    raise ValueError(f"not supported shape: {x.shape}")

def average_channel_variance(x):
  if x.ndim == 4:
    return x.var(dim=(0,2,3)).mean().item()
  elif x.ndim == 2:
    return x.var(dim=0).mean().item()
  else:
    raise ValueError(f"not supported shape: {x.shape}")

def extract_activations(model, *args, **kwargs):
  acts = []
  handles = []
  for name, module in model.named_modules():
    handle = module.register_forward_hook(partial(hook, name=name, store=acts))
    handles.append(handle)
  model(*args, **kwargs)
  for handle in handles:
    handle.remove()
  return acts

def hook(self, input, output, store=None, name=None):
  if store is None:
    store = []
  store.append((name, output))
