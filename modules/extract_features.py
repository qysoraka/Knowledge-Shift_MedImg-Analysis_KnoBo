
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision
import torchxrayvision as xrv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import clip
import open_clip
from utils import *
from models import DenseNetE2E
# from medclip import MedCLIPProcessor, MedCLIPModel, MedCLIPVisionModelViT # Please refer to the original MedCLIP repository to set up the environment: https://github.com/RyanWangZf/MedCLIP
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel

from argparse import ArgumentParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device being used:", device)