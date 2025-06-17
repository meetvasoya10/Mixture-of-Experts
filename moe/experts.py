import torch
import torch.nn as nn


class Expert(nn.Module):
    """
    Defines a single expert module used in the Mixture of Experts (MoE) framework.
    Each expert is a two-layer feed-forward neural network with GELU activation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()  # Matches GELU usage in GPT-Neo and Transformers
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the expert.

        Args:
            x: Tensor of shape [batch_size, input_dim]

        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x