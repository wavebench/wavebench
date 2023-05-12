""" LpLoss class for computing the relative error between two tensors. """
import torch

def lp_loss(input: torch.Tensor, target: torch.Tensor,
            p: int = 2, reduction: str = "mean"):
  batch_size = input.size(0)
  diff_norms = torch.norm(
    input.reshape(batch_size, -1) - target.reshape(batch_size, -1), p, 1)
  target_norms = torch.norm(
    target.reshape(batch_size, -1), p, 1)
  val = diff_norms / target_norms
  if reduction == "mean":
    return torch.mean(val)
  elif reduction == "sum":
    return torch.sum(val)
  elif reduction == "none":
    return val
  else:
    raise NotImplementedError(reduction)

class LpLoss(torch.nn.Module):
  """LpLoss for PDEs.

  Args:
      p (int, optional): p in Lp norm. Defaults to 2.
      reduction (str, optional): Reduction method. Defaults to "mean".
  """

  def __init__(self, p: int = 2, reduction: str = "mean") -> None:
    super().__init__()
    self.p = p
    self.reduction = reduction

  def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return lp_loss(input, target, p=self.p, reduction=self.reduction)
