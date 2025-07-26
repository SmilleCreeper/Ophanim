import torch
import streamlit as st
from typing import Dict
from projecting import project_embeddings

def convert_output_head(original_state_dict: Dict[str, torch.Tensor], converted_state_dict: Dict[str, torch.Tensor], st) -> None:
    """Handle output head (lm_head) conversion and dimension adjustment."""
    if "lm_head.weight" in original_state_dict:
        lm_head_weight = original_state_dict["lm_head.weight"]
        embed_weight = converted_state_dict.get("model.embed_tokens.weight")
        
        if embed_weight is not None:
            embed_dim = embed_weight.shape[1]
            
            # Check lm_head dimensions and fix if needed
            if len(lm_head_weight.shape) == 2:
                if lm_head_weight.shape[1] != embed_dim:
                    st.info(f"🔄 Adjusting lm_head dimensions: {lm_head_weight.shape} -> ({lm_head_weight.shape[0]}, {embed_dim})")
                    lm_head_weight = project_embeddings(lm_head_weight.T, embed_dim, st).T
            
        converted_state_dict["lm_head.weight"] = lm_head_weight