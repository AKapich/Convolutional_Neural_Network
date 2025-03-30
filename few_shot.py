"""
Reference: https://github.com/sicara/easy-few-shot-learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from easyfsl.samplers import TaskSampler


class PrototypicalNetwork(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
    ):
        super(PrototypicalNetwork, self).__init__()
        self.base_model = base_model

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.base_model.forward(support_images)
        z_query = self.base_model.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        scores = -torch.cdist(z_query, z_proto)
        return scores


class FewShotDataLoader:
    pass
