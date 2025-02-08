from typing import Optional

import torch

def linear_factory(in_features: int, out_features: int, bias: bool = True, device: Optional[torch.device] = None, dtype: torch.dtype = torch.bfloat16,
                   peft_type: Optional[str] = None, rank: Optional[int] = None, alpha: Optional[float] = None,
                   dropout: Optional[float] = None, ) -> torch.nn.Module:
    """
    Factory function for creating linear layers.
    """

    if peft_type is None:
        return torch.nn.Linear(in_features, out_features, bias, device, dtype)

    if peft_type not in ("lora", "dora"):
        raise ValueError(f"Invalid peft_type: {peft_type}.  Must be 'lora' or 'dora'.")

    if rank is None:
        raise ValueError(f"Rank must be specified when peft_type is '{peft_type}'")

    kwargs = {"in_dim": in_features, "out_dim": out_features, "rank": rank, "alpha": alpha, "dropout": dropout, "use_bias": bias}

    if peft_type == "lora":
        from torchtune.modules.peft import LoRALinear
        return LoRALinear(**kwargs)
    elif peft_type == "dora":
        from torchtune.modules.peft import DoRALinear
        return DoRALinear(**kwargs)

    raise ValueError(f"Should not reach here. Invalid peft_type: {peft_type}")