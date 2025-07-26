import torch
import streamlit as st
from typing import Dict

def convert_log_debug_info(original_state_dict: Dict[str, torch.Tensor], converted_state_dict: Dict[str, torch.Tensor], st) -> None:
    """Log debug information and check for unmapped tensors."""
    st.write(f"✅ Converted {len(converted_state_dict)} tensors")
    
    # Check for unmapped tensors (excluding RoPE inv_freq)
    original_keys = set(original_state_dict.keys())
    # Remove RoPE inverse frequency keys from original keys for comparison
    original_keys_filtered = {k for k in original_keys if not k.endswith('.rope.inv_freq')}
    
    # Create reverse mapping to check coverage
    converted_keys_mapped_back = set()
    for new_key in converted_state_dict.keys():
        # Map back to original naming convention
        if new_key == "model.embed_projection.weight":
            # Handle the special case of embedding projection
            converted_keys_mapped_back.add("embedding_projection.weight")
        else:
            old_key = new_key.replace("model.", "")
            old_key = old_key.replace("self_attn.", "attention.")
            old_key = old_key.replace("mlp.gate_proj", "feed_forward.w1")
            old_key = old_key.replace("mlp.down_proj", "feed_forward.w2")
            old_key = old_key.replace("mlp.up_proj", "feed_forward.w3")
            old_key = old_key.replace("input_layernorm", "norm1")
            old_key = old_key.replace("post_attention_layernorm", "norm2")
            converted_keys_mapped_back.add(old_key)
    
    unmapped = original_keys_filtered - converted_keys_mapped_back
    if unmapped:
        st.warning(f"⚠️ Unmapped tensors: {unmapped}")
    else:
        st.success("✅ All relevant tensors mapped successfully!")
    
    # Show some statistics
    st.write(f"📊 Original tensors: {len(original_keys)} (including {len(original_keys) - len(original_keys_filtered)} RoPE tensors)")
    st.write(f"📊 Converted tensors: {len(converted_state_dict)}")