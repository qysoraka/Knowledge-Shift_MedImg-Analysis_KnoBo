
import os
import json
import random
import wandb
import open_clip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
from utils import load_clip_model
random.seed(0)

# This script will fine-tune clip with the knowledge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# dataloader
class FinetuneDataset(Dataset):
    def __init__(self, data, image_dir, preprocess, tokenizer):
        self.data = data
        self.preprocess = preprocess
        self.image_paths = list(set([d[0] for d in data]))
        self.texts = list(set([d[1] for d in data]))

        print("Preprocessing images ...") # you need a lot of memory for this