""" LpLoss class for computing the relative error between two tensors. """
import torch

class LpLoss(object):
  """ LpLoss class for computing the relative error between two tensors. """
  def __init__(self, p=2, size_average=True, reduction=True):
    super(LpLoss, self).__init__()

    #Dimension and Lp-norm type are postive
    assert p > 0

    self.p = p
    self.reduction = reduction
    self.size_average = size_average

  def rel(self, x, y):
    num_examples = x.size()[0]
    diff_norms = torch.norm(
      x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
    y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

    if self.reduction:
      if self.size_average:
        return torch.mean(diff_norms/y_norms)
      else:
        return torch.sum(diff_norms/y_norms)

    return diff_norms/y_norms

  def __call__(self, x, y):
    return self.rel(x, y)