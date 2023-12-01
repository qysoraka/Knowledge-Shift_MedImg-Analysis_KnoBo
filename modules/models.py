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
        self.linear = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.vision_encoder(x)
        x = self.linear(x)
        return x


class MultiClassLogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes, prior, apply_prior=False):
        super(MultiClassLogisticRegression, self).__init__()
        self.linear = nn.Linear(num_features, num_classes)
        self.prior = prior.cuda() # prior is a weight matrix has the same shape as the weight matrix of the linear layer
        self.apply_prior = apply_prior
    
    def forward(self, x):
        return self.linear(x)


class PosthocHybridCBM(nn.Module):
    def __init__(self, n_concepts, n_classes, n_image_features, apply_prior=False):
        """
        PosthocCBM Hybrid Layer. 
        Takes an embedding as the input, outputs class-level predictions.
        Uses both the embedding and the concept predictions.
        Args:
            bottleneck (PosthocLinearCBM): [description]
        """
        super(PosthocHybridCBM, self).__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.n_image_features = n_image_features
        self.apply_prior = apply_prior

        self.bottleneck_classifier = nn.Linear(self.n_concepts, self.n_classes)
        self.residual_classifier = nn.Linear(self.n_image_features, self.n_classes)

    def forward(self, features):
        image_features = features[:, :self.n_image_features]
        concept_features = features[:, self.n_image_features:]

        out = self.bottleneck_classifier(concept_