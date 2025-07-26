import torch
from typing import Dict

def convert_normalization(original_state_dict: Dict[str, torch.Tensor], converted_state_dict: Dict[str, torch.Tensor]) -> None:
    """Handle output normalization conversion."""
    if "norm.weight" in original_state_dict:
        converted_state_dict["model.norm.weight"] = original_state_dict["norm.weight"]