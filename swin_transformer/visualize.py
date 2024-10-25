import numpy as np
import torch
import random
from typing import List,Dict
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from torchmetrics import ConfusionMatrix
#from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix
#from mlxtend.plotting import plot_confusion_matrix

import matplotlib.pyplot as plt

def visualize_images(dataloader, n=6):
  """Visualizes a specified number of images from a DataLoader.

  Args:
      dataloader (DataLoader): The DataLoader containing the images.
      n (int, optional): The number of images to visualize. Defaults to 6.
  """

  # Get a batch of images from the DataLoader
  images, labels = next(iter(dataloader))

  # Create a figure and subplots
  fig, axs = plt.subplots(1, n, figsize=(n * 4, 4))

  # Visualize each image
  for i in range(n):
      axs[i].imshow(images[i].permute(1, 2, 0))  # Permute dimensions for correct display
      axs[i].set_title(f"Label: {labels[i]}")
      axs[i].axis('off')

  plt.show()
  
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes:  bool = True,
                          n: int = 5,
                          display_shape: bool = False):
  if n > 5:
      n = 5
      display_shape = False
      print(f"For display purposes, n shouldn't be larger than 5, setting to 5 and removing shape display.")

  random_samples_idx = random.sample(range(len(dataset)), k=n)
  print(f"Format of labels: {dataset.attribs}")
  plt.figure(figsize=(16, 10))

  for i, targ_sample in enumerate(random_samples_idx):
    targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
    targ_label = dataset.decode_labels_onehot(targ_label)
    targ_image_adjust = targ_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    plt.subplot(1, n, i+1)
    plt.imshow(targ_image_adjust)
    plt.axis("off")
    plt.rc('axes', titlesize=8)
    if classes:
        title = list(targ_label)
        if display_shape:
            title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)    

def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
