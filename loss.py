import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class OCSoftmax(nn.Module):
    def __init__(
        self,
        feat_dim=2,
        m_real=0.5,
        m_fake=0.2,
        alpha=20.0,
        fix_centers=True,
        initialize_centers="one_hot",
    ):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha
        if initialize_centers == "one_hot":
            self.center = nn.Parameter(
                torch.eye(self.feat_dim)[:1], requires_grad=not fix_centers
            )
        elif initialize_centers == "random":
            self.center = nn.Parameter(
                torch.randn(1, self.feat_dim), requires_grad=not fix_centers
            )
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0, 1)
        output_scores = scores.clone()

        scores[labels == 0] = self.m_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.m_fake

        loss = self.softplus(self.alpha * scores).mean()

        return loss, output_scores.squeeze(1)
