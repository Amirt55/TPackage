import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix, Accuracy, Precision, Recall, F1Score, Specificity, AUROC, PrecisionRecallCurve
import seaborn as sns
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
import tqdm
import logging
import sklearn
import os, shutil, copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image



def show_random_samples(dataframe, num_samples=9):
  """
  *** Showing Random Samples ***
  Randomly sample the specified number of rows from the dataframe

  Argument(s):
  -dataframe - Pandas DataFrame

  -num_samples - int: Optionally, Set the number of samples to show
  """
  sample_df = dataframe.sample(n=num_samples)

  plt.figure(figsize=(10, 10))
  for idx, row in enumerate(sample_df.itertuples(), 1):
    image_path = row.path
    target = row.target
    location = row.location

    image = Image.open(image_path)

    plt.subplot(3, 3, idx)
    plt.imshow(image, cmap='gray')
    target = 'Normal' if target == 0 else 'Pneumonia'
    plt.title(f"Class: {target}\nSet: {location}")
    plt.axis('off')

  plt.tight_layout()
  plt.show()


def visualizer(dataframe, *plot_type):
  """
  *** Visualizing the Dataframe ***
  The function for analyzing and visualizing dataframes in various types

  Argument(s):
  -dataframe - Pandas DataFrame

  -plot_type - str: Set the type or types of visualizing
  """
  colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
  for vis in plot_type:
    if vis == 'hist':
      counts, _, _= plt.hist(dataframe.location, bins=len(dataframe.location.unique()))
      plt.xlabel(f'{dataframe.location.unique()[0]}:{counts[0]}, {dataframe.location.unique()[1]}:{counts[1]}, {dataframe.location.unique()[2]}:{counts[2]}')
      plt.ylabel('Count')
      plt.show()
      print('-'*10, end='\n')

    if vis == 'distbar':
      dataframe['target'].value_counts().plot(kind='bar')
      plt.xlabel('Target')
      plt.ylabel('Count')
      plt.title('Distribution of Target Classes')
      plt.show()
      print('-'*10, end='\n')

    if vis == 'dfadv':
      fig, axes = plt.subplots(1, 3, figsize=(15, 5))
      for idx, phase in enumerate(dataframe['location'].unique()):
        subplot_dataframe_dist(colors, dataframe, phase, axes[idx])
      plt.tight_layout()
      plt.show()
      print('-'*10, end='\n')

    if vis == 'dfsimp':
      fig, axes = plt.subplots(1, 3, figsize=(15, 5))
      for idx in range(len(dataframe.location.unique())):
        subplot_dataframe(idx, dataframe, dataframe.location.unique()[idx], axes)
      plt.tight_layout()
      plt.show()
      print('-'*10, end='\n')

    if vis == 'viobox':
      plt.figure(figsize=(12, 6))
      plt.boxplot([dataframe.target[dataframe.location == st] for st in dataframe.location.unique()], vert=False, labels=dataframe.location.unique())
      plt.violinplot([dataframe.target[dataframe.location == st] for st in dataframe.location.unique()], vert=False)
      plt.show()
      print('-'*10, end='\n')


def subplot_dataframe_dist(colors, df, phase, ax):
  temp = df[df['location'] == phase]['target']
  unique_targets = temp.unique()
  # bins = np.arange(len(unique_targets) + 1) - 0.5  # Define bins to align with the unique targets

  for i, target in enumerate(unique_targets):
    bin_color = colors[i % len(colors)]  # Cycle through colors if there are more bins than colors
    count = (temp == target).sum()
    ax.bar(target, count, color=bin_color, edgecolor='black', label=f'Target {target}')

  ax.set_title(phase.capitalize())
  ax.set_xlabel('Target')
  ax.set_ylabel('Count')
  ax.legend()


def subplot_dataframe(idx, df, phase, axes):
  temp = df[df['location'] == phase]['target']
  axes[idx].hist(temp, bins=len(temp.unique()))
  axes[idx].set_title(phase.capitalize())
  axes[idx].set_xlabel('Target')
  axes[idx].set_ylabel('Count')



class NormMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.mean = 0
    self.std = 0

  def update(self, sum_pixel, sum_suare_pixel, num_pixel):
    self.mean = sum_pixel / num_pixel
    self.std = np.sqrt((sum_suare_pixel / num_pixel) - (self.mean ** 2))


def calculate_mean_std(df, data_type='train'):
  """
  *** Mean-Std Calculator ***
  The function for Calculating mean and std of images in dataframe

  Argument(s):
  -df - Pandas DataFrame

  -data_type - str: Optionally, Set the type of data
  ('train', 'val' or 'test')

  Return(s):
  -norms.mean - float: TrainSet Mean
  -norms.std - float: TrainSet Mean
  """
  sum_pixel_values = 0
  sum_pixel_squares = 0
  num_pixels = 0
  norms = NormMeter()

  subset_df = df[df['location'] == data_type]
  with tqdm.tqdm(subset_df['path'], unit='image') as tqdms:
    for image_path in tqdms:
      tqdms.set_description('Trainset-Normimg')
      image = Image.open(image_path).convert('L')
      img_array = np.array(image).astype(np.float32) / 255.0
      sum_pixel_values += np.sum(img_array)
      sum_pixel_squares += np.sum(np.square(img_array))
      num_pixels += img_array.size
      norms.update(sum_pixel_values, sum_pixel_squares, num_pixels)
      tqdms.set_postfix(Mean=norms.mean, Std=norms.std)
  print(f"\nTraining Set Mean: {norms.mean}, Std: {norms.std}")
  return norms.mean, norms.std


def set_seed(seed, cuda_seed=False, multi_gpu=False, cudnn_state=None):
  """
  *** Set random seed for reproducibility ***
  This ensures that all operations are deterministic and can be
  reproduced across runs and machines.

  Argument(s):
  -seed - int: The random seed to use

  -cuda_seed - boolean: Optionally, Set the seed for CUDA
  operations

  -multi_gpu - boolean: Optionally, Set additional flags
  for reproducibility in multi-threaded or multi-GPU
  settings

  -cudnn_state - list[deterministic, benchmark]: Optionally
  Set deterministic behavior for cuDNN (GPU acceleration library)
  """
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  if cuda_seed and torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

  if cudnn_state is not None:
    torch.backends.cudnn.deterministic = cudnn_state[0] # Ensure deterministic algorithms: True
    torch.backends.cudnn.benchmark = cudnn_state[1] # Disable cuDNN's auto-tuning for algorithms: False

  if multi_gpu:
    torch.set_deterministic(True) # Can be used in newer versions for stricter reproducibility



def normalize_image(image, eps):
  image_min = image.min()
  image_max = image.max()
  image.clamp_(min = image_min, max = image_max)
  image.add_(-image_min).div_(image_max - image_min + eps)
  return image


def show_batch_samples(images, labels, classes, normalize=True):
  """
  *** Plotting Samples in Batch ***
  Plot samples of a batch with their labels

  Argument(s):
  -images - batch-iter[0]

  -labels - batch-iter[1]

  -classes - trainset class

  -normalize - boolean: Optionally
  """
  n_images = len(images)

  rows = int(np.sqrt(n_images))
  cols = int(np.sqrt(n_images))

  fig = plt.figure(figsize=(20, 20))

  for i in range(rows*cols):
    ax = fig.add_subplot(rows, cols, i+1)

    image = images[i]

    if normalize: image = normalize_image(image, 1e-5)

    ax.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='bone')
    ax.set_title(classes[int(labels[i])])
    ax.axis('off')


def show_batch_samples_with_preds(images, labels, preds, classes, normalize=False):
  """
  *** Plotting Samples in Batch with Model Predictions***
  Plot samples of a batch with their labels (GroundTruth) 
  and model predictions

  Argument(s):
  -images - batch-iter[0]

  -labels - batch-iter[1]

  -preds - as(labels): Model Outputs

  -classes - Testset class

  -normalize - boolean: Optionally
  """
  n_images = len(images)

  rows = int(np.sqrt(n_images))
  cols = int(np.sqrt(n_images))

  fig = plt.figure(figsize=(30, 30))

  for i in range(rows*cols):
    ax = fig.add_subplot(rows, cols, i+1)

    image = images[i]

    if normalize: image = normalize_image(image, 1e-5)

    ax.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='bone')
    ax.set_title(f"True: {classes[int(labels[i].cpu())]}, Pred: {classes[int(preds[i].cpu())]}")
    ax.axis('off')



def num_trainable_params(model):
  """
  *** Calculating the Number of Trainable Parameters ***
  The function returns the number of trainable parameters of
  the model in billion scale.

  Argument(s):
  -model - pytorch nn.Module class: Model Structure

  Return(s):
  -nums/1000000 - float
  """
  nums = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return nums/1000000


def empty_gpu(sch=True):
  """
  *** Freeing up GPU space ***
  The model, optimizer, scheduler (if needed), batches and
  outputs of the model must be defined before.

  Argument(s):
  -sch - boolean: LR_Scheduler
  """
  torch.cuda.empty_cache()
  temp = torch.tensor(1, device=device)
  model = temp.clone()
  optimizer = temp.clone()
  if sch: scheduler = temp.clone()
  inputs = temp.clone()
  targets = temp.clone()
  outputs = temp.clone()



class AverageMeter(object):
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


def train(epoch, train_loader, options=[True, False, True, False, None]):
  """
  *** One Epoch Training Function ***
  This function trains the model along with storing the metrics
  and plotting the related curves in TensorBoard.

  Argument(s):
  -epoch - int: Epoch number

  -train_loader - DataLoader: Train Dataloader in PyTorch format

  -options - list[TensorBoard, Activation_Function, LR_Scheduler,
  Small_Train, Small_Train_Batch_Number]: Optionally, Set some
  Options for Modeling and Visualizing

  Return(s):
  -model - pytorch nn.Module class: Model Structure
  """
  model.train()

  loss_train = AverageMeter()

  accuracy = Accuracy(task="binary", num_classes=2).to(device)
  precision = Precision(task="binary", num_classes=2).to(device)
  recall = Recall(task="binary", num_classes=2).to(device)
  f1_score = F1Score(task="binary", num_classes=2).to(device)
  specificity = Specificity(task="binary", num_classes=2).to(device)
  auc = AUROC(task="binary").to(device)

  with tqdm.tqdm(train_loader, unit='batch') as trainpock:
    for batch_idx, (inputs, targets) in enumerate(trainpock):
      trainpock.set_description(f'Train: Epoch {epoch}')

      inputs = inputs.to(device)
      targets = targets.to(device)

      optimizer.zero_grad()

      outputs = model(inputs)
      if options[1]: outputs = torch.sigmoid(outputs)

      loss = loss_fn(outputs, targets)
      loss.backward()

      optimizer.step()

      loss_train.update(loss.item(), n=len(targets))

      outputs = torch.round(outputs)
      accuracy(outputs, targets)
      precision(outputs, targets)
      recall(outputs, targets)
      f1_score(outputs, targets)
      specificity(outputs, targets)
      auc(outputs, targets)

      if options[3] and batch_idx == options[4]: break

      if options[2]:
        lrsch_list_per_step.append(scheduler.get_last_lr()[0])
        scheduler.step()
        trainpock.set_postfix(loss=loss_train.avg, LR=scheduler.get_last_lr()[0])
      else:
        trainpock.set_postfix(loss=loss_train.avg)

  accuracy_temp = accuracy.compute()
  precision_temp = precision.compute()
  recall_temp = recall.compute()
  f1_score_temp = f1_score.compute()
  specificity_temp = specificity.compute()
  auc_temp = auc.compute()

  loss_train_list.append(loss_train.avg)
  if options[2]:
    lrsch_list_per_epoch.append(scheduler.get_last_lr()[0])
  accuracy_list.append(accuracy_temp.item())
  precision_list.append(precision_temp.item())
  recall_list.append(recall_temp.item())
  f1_score_list.append(f1_score_temp.item())
  specificity_list.append(specificity_temp.item())
  auc_list.append(auc_temp.item())

  if options[0]:
    writer.add_scalar('Loss/train', loss_train.avg, epoch)
    writer.add_scalar('Acc/train', accuracy_temp.item(), epoch)
    writer.add_scalar('Prc/train', precision_temp.item(), epoch)
    writer.add_scalar('Rec/train', recall_temp.item(), epoch)
    writer.add_scalar('F1/train', f1_score_temp.item(), epoch)
    writer.add_scalar('Spc/train', specificity_temp.item(), epoch)
    writer.add_scalar('AUC/train', auc_temp.item(), epoch)
    
  if options[2]:
    logger.info(f'Train: Epoch:{epoch} LR:{scheduler.get_last_lr()[0]:.4} Loss:{loss_train.avg:.4} Accuracy:{accuracy_temp.item():.4}')
  else:
    logger.info(f'Train: Epoch:{epoch} Loss:{loss_train.avg:.4} Accuracy:{accuracy_temp.item():.4}')

  accuracy.reset()
  precision.reset()
  recall.reset()
  f1_score.reset()
  specificity.reset()
  auc.reset()

  return model


def plot_specific_metrics(metric_type, prcurve=None, confusion=None):
  if metric_type == 'roccurve':
    plt.figure(figsize=(6, 6))
    temp = 1-prcurve[0].cpu().numpy()
    plt.plot(temp,
             prcurve[1].cpu().numpy(), color='b', lw=2)
    plt.xlabel('False Positive Rate (1-Specifity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve')
    plt.grid(True)
    print()

  if metric_type == 'prcurve':
    plt.figure(figsize=(6, 6))
    plt.plot(prcurve[0].cpu().numpy(),
             prcurve[1].cpu().numpy(), color='b', lw=2)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    print()

  if metric_type == 'confusion':
    confusion = confusion.cpu().numpy()
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion,
                annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    print('[TN, FP]\n[FN, TP]')
    plt.title('Confusion Matrix')
    print()

  if metric_type == 'all':
    plot_specific_metrics(metric_type='confusion', confusion=confusion)
    plot_specific_metrics(metric_type='roccurve', prcurve=prcurve)
    plot_specific_metrics(metric_type='prcurve', prcurve=prcurve)

  plt.show()


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename, scheduler=None, save_rng_state=False, outret=False):
  """
  *** Saving Chechpoint ***
  This function saves model weights, optimizer state, LR scheduler
  state if it's define before and RNG state if needed.

  Argument(s):
  -model - pytorch nn.Module class: Model Structure

  -optimizer - pytorch optim

  -epoch - int: Epoch number

  -loss - float: The value of loss

  -accuracy - float: The value of accuracy

  -filename - str: Chechpoint Directory

  -scheduler - pytorch optim lr_scheduler

  -save_rng_state - boolean: If True, the RNG-states are saved.

  -outret - boolean: If True, the Ckeckpoint dictionary are
  retruned.
  """
  logger.info(f"Saving checkpoint at epoch {epoch}...")
  ckpts_dir = 'ckpts/'
  os.makedirs(ckpts_dir, exist_ok=True)

  checkpoint = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': loss,
      'accuracy': accuracy
  }

  if save_rng_state:
    logger.info(f"Saving rng-states at checkpoint...")
    checkpoint['rng_state'] = torch.get_rng_state() # Save RNG state for reproducibility
    checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

  if scheduler is not None:
    logger.info(f"Saving scheduler at checkpoint...")
    checkpoint['scheduler_state_dict'] = scheduler.state_dict()

  torch.save(checkpoint, ckpts_dir + filename)
  logger.info(f"Checkpoint saved!")

  if outret:
    return checkpoint


def resume_checkpoint(model, filename, cat=[True, False, None, None, False], outret=False):
  """
  *** Loading Chechpoint ***
  This function loads model weights, optimizer state, LR scheduler
  state if it's define before and RNG state if needed.

  Argument(s):
  -model - pytorch nn.Module class: Model Structure

  -filename - str: Chechpoint Directory

  -cat - list[strict, map_location, optimizer, scheduler, load_rng_state]:
  Optionally, Set strict and map_location booleans for exact matching,
  optimizer - pytorch optim, Scheduler - pytorch optim lr_scheduler and
  load_rng_state - boolean to load the RNG state if needed

  -outret - boolean: If True, the Ckeckpoint dictionary are
  retruned.

  Return(s):
  -model - pytorch nn.Module class: Model Structure
  -epoch - int: Epoch number where the model was last saved
  -loss - float: The loss at which the model was last saved
  -accuracy - float: The accuracy at which the model was last saved
  """
  if not os.path.exists(filename):
    logging.error(f"Checkpoint '{filename}' not found!")
    return 0, 0, 0.0, 0.0  # Return 0 so training can start from scratch

  if cat[1]:
    checkpoint = torch.load(filename, weights_only=False, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
  else:
    checkpoint = torch.load(filename, weights_only=False)

  model.load_state_dict(checkpoint['model_state_dict'], strict=cat[0])

  if cat[2] is not None:
    cat[2].load_state_dict(checkpoint['optimizer_state_dict'])
  if cat[3] is not None:
    cat[3].load_state_dict(checkpoint['scheduler_state_dict'])

  epoch = checkpoint['epoch']
  loss = checkpoint['loss']
  accuracy = checkpoint['accuracy']

  if cat[4]:
    torch.set_rng_state(checkpoint['rng_state'])
    if torch.cuda.is_available() and checkpoint['cuda_rng_state'] is not None:
      torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])

  logging.info(f"Checkpoint at epoch {epoch} Loaded.")
  
  if outret:
    return checkpoint
  else:
    return model, epoch, loss, accuracy


def inference(model, loader, epoch=None, options=[True, False, True, None]):
  """
  *** One Epoch Validation and Inference Function ***
  This function validates the model along with Plotting the related
  curves in TensorBoard and the specific metrics if needed.

  Argument(s):
  -model - pytorch nn.Module class: Model Structure with its weights

  -loader - DataLoader: Valid or Test Dataloader in PyTorch format

  -epoch - int: Optionally, Function in validation mode if epoch
  number has been defined; else, Inference mode
  
  -options - list[TensorBoard, Activation_Function, LR_Scheduler,
  Plot_Specific_Metrix]: Optionally, Set some Options for Modeling
  and Visualizing
  """
  model.eval()

  loss_eval = AverageMeter()

  accuracy = Accuracy(task="binary", num_classes=2).to(device)
  precision = Precision(task="binary", num_classes=2).to(device)
  recall = Recall(task="binary", num_classes=2).to(device)
  f1_score = F1Score(task="binary", num_classes=2).to(device)
  specificity = Specificity(task="binary", num_classes=2).to(device)
  auc = AUROC(task="binary").to(device)

  confusion_matrix = ConfusionMatrix(task="binary", num_classes=2).to(device)
  pr_curve = PrecisionRecallCurve(task="binary").to(device)

  with torch.no_grad():
    with tqdm.tqdm(loader, unit='batch') as infpock:
      for inputs, targets in infpock:
        if epoch is not None:
          infpock.set_description(f'Valid: Epoch {epoch}')
        else:
          infpock.set_description(f'Inference:')

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        if options[1]: outputs = torch.sigmoid(outputs)

        loss = loss_fn(outputs, targets)

        loss_eval.update(loss.item(), n=len(targets))

        outputs = torch.round(outputs)
        accuracy(outputs, targets)
        precision(outputs, targets)
        recall(outputs, targets)
        f1_score(outputs, targets)
        specificity(outputs, targets)
        auc(outputs, targets)
        confusion_matrix(outputs, targets)
        pr_curve(outputs, targets.long())

        infpock.set_postfix(loss=loss_eval.avg)

    accuracy_temp = accuracy.compute()
    precision_temp = precision.compute()
    recall_temp = recall.compute()
    f1_score_temp = f1_score.compute()
    specificity_temp = specificity.compute()
    auc_temp = auc.compute()

    confusion_matrix_temp = confusion_matrix.compute()

    loss_eval_list.append(loss_eval.avg)
    accuracy_eval_list.append(accuracy_temp.item())
    precision_eval_list.append(precision_temp.item())
    recall_eval_list.append(recall_temp.item())
    f1_score_eval_list.append(f1_score_temp.item())
    specificity_eval_list.append(specificity_temp.item())
    auc_eval_list.append(auc_temp.item())

    confusion_matrix_list.append(confusion_matrix_temp)
    # pr_curve_list.append([pr_curve.compute()])

    if options[0] and epoch is not None:
      writer.add_scalar('Loss/valid', loss_eval.avg, epoch)
      writer.add_scalar('Acc/valid', accuracy_temp.item(), epoch)
      writer.add_scalar('Prc/valid', precision_temp.item(), epoch)
      writer.add_scalar('Rec/valid', recall_temp.item(), epoch)
      writer.add_scalar('F1/valid', f1_score_temp.item(), epoch)
      writer.add_scalar('Spc/valid', specificity_temp.item(), epoch)
      writer.add_scalar('AUC/valid', auc_temp.item(), epoch)
    logger.info(f'Inference - Loss: {loss_eval.avg:.4f}, '
                f'Accuracy: {accuracy_temp.item():.4f}, Precision: {precision_temp.item():.4f}, '
                f'Recall: {recall_temp.item():.4f}, F1: {f1_score_temp.item():.4f}, '
                f'Specificity: {specificity_temp.item():.4f}, AUC: {auc_temp.item():.4f}')
    logger.info(f'Confusion Matrix: \n{confusion_matrix_temp}')

    if options[3] is not None:
      if options[3] == 'prcurve' or options[3] == 'roccurve':
        plot_specific_metrics(metric_type=options[3], prcurve=pr_curve.compute())
      elif options[3] == 'confusion':
        plot_specific_metrics(metric_type=options[3], confusion=confusion_matrix_temp)
      elif options[3] == 'all':
        plot_specific_metrics(metric_type=options[3], prcurve=pr_curve.compute(), confusion=confusion_matrix_temp)
      else:
        logger.error('Invalid input for plotting argument!')
    else:
      logger.warning('Plot specific metrics off!')

    accuracy.reset()
    precision.reset()
    recall.reset()
    f1_score.reset()
    specificity.reset()
    auc.reset()
    confusion_matrix.reset()
    pr_curve.reset()

  if epoch is not None:
    filename = f'checkpoint_{epoch}.pth'
    if options[2]:
      save_checkpoint(model, optimizer, epoch, loss_eval.avg, accuracy_temp.item(), filename, scheduler)
    else:
      save_checkpoint(model, optimizer, epoch, loss_eval.avg, accuracy_temp.item(), filename)
