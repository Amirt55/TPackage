import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import tqdm
import torchmetrics as tm


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def num_trainable_params(model):
  nums = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return nums


def train_one_epoch(model, train_loader, loss_fn, optimizer, metric, epoch=0, device='cpu'):
  model.train()
  loss_train = AverageMeter()
  metric.reset()
  with tqdm.tqdm(train_loader, unit='batch') as tepoch:
    for inputs, targets in tepoch:
      tepoch.set_description(f'Epoch {epoch}')# if epoch
      inputs = inputs.to(device)
      targets = targets.to(device)
      outputs = model(inputs)
      loss = loss_fn(outputs, targets)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      loss_train.update(loss.item(), n=len(targets))
      metric.update(outputs, targets)
      tepoch.set_postfix(loss=loss_train.avg, metric=metric.compute().item())
  return model, loss_train.avg, metric.compute().item()


def evaluate(model, test_loader, loss_fn, metric, num_val=0, device='cpu'):
  model.eval()
  loss_eval = AverageMeter()
  metric.reset()
  with torch.no_grad():
    with tqdm.tqdm(test_loader, unit='batch') as testpoch:
      for inputs, targets in testpoch:
        testpoch.set_description(f'Valid {num_val}')
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss_eval.update(loss.item(), n=len(targets))
        metric(outputs, targets)
        testpoch.set_postfix(loss=loss_eval.avg, metric=metric.compute().item())
  return loss_eval.avg, metric.compute().item()
