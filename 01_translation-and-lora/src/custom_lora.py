import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    Manual implementation of Low-Rank Adaptation (LoRA).
    Wraps an existing linear layer to inject trainable rank-decomposition matrices.
    """

    def __init__(self, module: nn.Linear, rank: int):
        super().__init__()
        self.module = module
        self.adapter_A = nn.Parameter(
            torch.empty(module.in_features, rank, device=module.weight.device)
        )
        nn.init.kaiming_uniform_(self.adapter_A, a=5**0.5)
        self.adapter_B = nn.Parameter(
            torch.zeros(rank, module.out_features, device=module.weight.device)
        )

    def forward(self, input):
        # base model output is frozen
        base_out = self.module(input)

        # adapter output is trainable
        adapter_out = (input @ self.adapter_A) @ self.adapter_B
        return base_out + adapter_out
