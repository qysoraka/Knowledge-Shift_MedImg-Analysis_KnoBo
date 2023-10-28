import torch
import torch.nn as nn

class LogisticRegressionT(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression