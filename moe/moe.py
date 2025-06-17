import torch
import torch.nn as nn

from experts import Expert
from gating import GatingNetwork


class MoE(nn.Module):
    """
    Mixture of Experts (MoE) model.

    Dynamically routes each token representation to a weighted combination of experts
    using a learned gating mechanism.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_experts: int = 3,
        expert_hidden_dim: int = 4096
    ):
        """
        Args:
            hidden_size (int): Input/output dimensionality of each token.
            num_experts (int): Number of expert networks.
            expert_hidden_dim (int): Hidden layer size inside each expert.
        """
        super().__init__()
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            Expert(hidden_size, expert_hidden_dim, hidden_size)
            for _ in range(num_experts)
        ])
        self.gate = GatingNetwork(hidden_size, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MoE layer.

        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].

        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        batch_size, seq_len, hidden_size = x.shape

        # Flatten sequence and batch for gating and expert processing
        x_flat = x.view(-1, hidden_size)

        # Compute gating weights and expert outputs
        gate_output = self.gate(x_flat)  # [B*S, E]
        expert_outputs = torch.stack(
            [expert(x_flat) for expert in self.experts], dim=2
        )  # [B*S, H, E]

        # Weighted sum of expert outputs
        output = torch.einsum('be,bed->bd', gate_output, expert_outputs)

        return output.view(batch_size, seq_len, hidden_size)