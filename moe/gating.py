import torch
import torch.nn as nn


class GatingNetwork(nn.Module):
    """
    Gating network used in Mixture of Experts (MoE).
    Computes softmax scores over experts for input routing.
    """

    def __init__(self, input_dim: int, num_experts: int):
        """
        Args:
            input_dim (int): Size of the input feature vector.
            num_experts (int): Total number of expert models.
        """
        super().__init__()
        self.layer = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute routing weights for experts.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Softmax probabilities of shape [batch_size, num_experts].
        """
        return torch.softmax(self.layer(x), dim=-1)