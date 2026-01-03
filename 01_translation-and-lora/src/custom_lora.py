import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, module: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.module = module
        self.scaling = alpha / rank
        
        self.module.requires_grad_(False)
        
        self.adapter_A = nn.Parameter(
            torch.empty(module.in_features, rank, device=module.weight.device)
        )
        nn.init.kaiming_uniform_(self.adapter_A, a=5**0.5)
        self.adapter_B = nn.Parameter(
            torch.zeros(rank, module.out_features, device=module.weight.device)
        )

    def forward(self, input):
        base_out = self.module(input)
        adapter_out = (input @ self.adapter_A) @ self.adapter_B
        return base_out + (adapter_out * self.scaling)