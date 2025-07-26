import torch
import streamlit as st
from typing import Dict
from model_config import ModelConfig


def convert_layer_weights(model_config: ModelConfig, original_state_dict: Dict[str, torch.Tensor],
                          converted_state_dict: Dict[str, torch.Tensor], include_debug_info: bool, st) -> None:
    """Convert layer weights including attention, feed-forward, and normalization."""
    for layer_idx in range(model_config.num_layers):
        # Attention projections
        attention_mappings = {
            f"layers.{layer_idx}.attention.q_proj.weight": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"layers.{layer_idx}.attention.k_proj.weight": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            f"layers.{layer_idx}.attention.v_proj.weight": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            f"layers.{layer_idx}.attention.o_proj.weight": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
        }

        for old_name, new_name in attention_mappings.items():
            if old_name in original_state_dict:
                converted_state_dict[new_name] = original_state_dict[old_name]

        # ВАЖНО: Сохраняем RoPE параметры в формате Llama
        rope_key = f"layers.{layer_idx}.attention.rope.inv_freq"
        if rope_key in original_state_dict:
            # Llama ожидает rotary_emb.inv_freq в каждом слое
            new_rope_key = f"model.layers.{layer_idx}.self_attn.rotary_emb.inv_freq"
            converted_state_dict[new_rope_key] = original_state_dict[rope_key]
            if include_debug_info:
                st.write(f"✅ Saved RoPE: {rope_key} -> {new_rope_key}")

        # Feed-forward projections - БЕЗ транспозиции!
        ffn_mappings = [
            (f"layers.{layer_idx}.feed_forward.w1.weight", f"model.layers.{layer_idx}.mlp.gate_proj.weight", False),
            (f"layers.{layer_idx}.feed_forward.w2.weight", f"model.layers.{layer_idx}.mlp.up_proj.weight", False),
            (f"layers.{layer_idx}.feed_forward.w3.weight", f"model.layers.{layer_idx}.mlp.down_proj.weight", False),
        ]

        for old_name, new_name, should_transpose in ffn_mappings:
            if old_name in original_state_dict:
                tensor = original_state_dict[old_name].clone()
                converted_state_dict[new_name] = tensor

                if include_debug_info:
                    st.write(f"✅ Mapped {old_name} -> {new_name}: {tensor.shape}")

        # Layer normalization
        norm_mappings = {
            f"layers.{layer_idx}.norm1.weight": f"model.layers.{layer_idx}.input_layernorm.weight",
            f"layers.{layer_idx}.norm2.weight": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
        }

        for old_name, new_name in norm_mappings.items():
            if old_name in original_state_dict:
                converted_state_dict[new_name] = original_state_dict[old_name]