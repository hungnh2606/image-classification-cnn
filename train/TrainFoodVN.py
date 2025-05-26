import os.path
import shutil

import cv2
import torch
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import Compose, Resize, ToTensor, RandomAffine, ColorJitter
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from argparse import ArgumentParser
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_transform = Compose([

    ])