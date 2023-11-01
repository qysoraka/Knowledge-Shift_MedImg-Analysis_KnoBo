import torch
import torch.nn as nn

class LogisticRegressionT(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionT, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class MultiLabelModel(nn.Module):
    def __init__(self, model, num_classes):
        super(MultiLabelModel, self).__init__()
        self.num_classes = num_classes
        self.vision_encoder = model.visual
        self.linear = nn.Line