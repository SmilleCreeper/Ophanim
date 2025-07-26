# pages/gguf_export.py - FIXED VERSION

import streamlit as st
import torch
import json
import os
import tempfile
import hashlib
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import safetensors.torch
import math


class ModelConfig:
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size, num_key_value_heads=None):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.intermediate_size = intermediate_size
        self.rope_theta = 10000.0
        self.max_position_embeddings = 2048
        self.head_dim = hidden_size // num_heads

    def create_hf_config(self, tokenizer, use_f32):
        return {
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 1,
            "eos_token_id": tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 2,
            "hidden_act": "silu",
            "hidden_size": self.hidden_size,
            "initializer_range": 0.02,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "model_type": "llama",
            "num_attention_heads": self.num_heads,
            "num_hidden_layers": self.num_layers,
            "num_key_value_heads": self.num_key_value_heads,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-5,
            "rope_scaling": None,
            "rope_theta": self.rope_theta,
            "tie_word_embeddings": True,
            "torch_dtype": "float32" if use_f32 else "float16",
            "transformers_version": "4.36.0",
            "use_cache": True,
            "vocab_size": self.vocab_size
        }


def extract_model_config(model):
    """Extract model configuration from the actual model tensors"""
    state_dict = model.state_dict()
    
    # Get dimensions from actual tensors
    hidden_size = state_dict["model.model.embed_tokens.weight"].shape[1]
    vocab_size = state_dict["model.model.embed_tokens.weight"].shape[0]
    
    # Count layers
    num_layers = len([k for k in state_dict.keys() if k.startswith("model.model.layers.") and k.endswith(".input_layernorm.weight")])
    
    # Determine head configuration from attention weights
    q_proj_out_dim = state_dict["model.model.layers.0.self_attn.q_proj.weight"].shape[0]
    k_proj_out_dim = state_dict["model.model.layers.0.self_attn.k_proj.weight"].shape[0]
    
    head_dim = 64  # Standard for Llama
    num_attention_heads = q_proj_out_dim // head_dim
    num_key_value_heads = k_proj_out_dim // head_dim
    
    # Get MLP intermediate size
    intermediate_size = state_dict["model.model.layers.0.mlp.gate_proj.weight"].shape[0]
    
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size
    )


def generate_rope_freqs(model_config):
    """Generate RoPE frequencies tensor in the format llama.cpp expects"""
    head_dim = model_config.head_dim
    theta = model_config.rope_theta
    
    # llama.cpp expects just the base frequencies, not the full position encodings
    # This should be a 1D tensor with head_dim/2 elements
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    
    return freqs


def convert_embeddings(model, original_state_dict, converted_state_dict):
    """Convert embedding weights"""
    if "model.model.embed_tokens.weight" in original_state_dict:
        converted_state_dict["model.embed_tokens.weight"] = original_state_dict["model.model.embed_tokens.weight"]
    else:
        # Fallback to any key with "embed" in it
        for key in original_state_dict.keys():
            if "embed" in key.lower():
                converted_state_dict["model.embed_tokens.weight"] = original_state_dict[key]
                break


def convert_normalization(original_state_dict, converted_state_dict):
    """Convert normalization layer"""
    if "model.model.norm.weight" in original_state_dict:
        converted_state_dict["model.norm.weight"] = original_state_dict["model.model.norm.weight"]
    else:
        # Fallback
        for key in original_state_dict.keys():
            if key.endswith("norm.weight") and "layers." not in key:
                converted_state_dict["model.norm.weight"] = original_state_dict[key]
                break


def convert_output_head(original_state_dict, converted_state_dict):
    """Convert output head (lm_head) - FIX: Remove 'output.weight', use correct mapping"""
    # Look for lm_head - your model uses model.lm_head.weight
    if "model.lm_head.weight" in original_state_dict:
        converted_state_dict["lm_head.weight"] = original_state_dict["model.lm_head.weight"]
        st.info("✅ Found and mapped model.lm_head.weight -> lm_head.weight")
    else:
        # Search for any lm_head variant
        found_lm_head = False
        for key in original_state_dict.keys():
            if "lm_head" in key.lower():
                converted_state_dict["lm_head.weight"] = original_state_dict[key]
                found_lm_head = True
                st.info(f"✅ Found and mapped {key} -> lm_head.weight")
                break
        
        # If no lm_head found, tie with embeddings (common in Llama)
        if not found_lm_head and "model.embed_tokens.weight" in converted_state_dict:
            converted_state_dict["lm_head.weight"] = converted_state_dict["model.embed_tokens.weight"].clone()
            st.info("✅ Tied lm_head.weight with embeddings (weight tying)")
    
    # IMPORTANT: Do NOT create "output.weight" - this is not part of standard Llama architecture


def convert_layer_weights(model_config, original_state_dict, converted_state_dict):
    """Convert layer weights using the actual tensor names from your model"""
    
    for layer_idx in range(model_config.num_layers):
        hf_layer_prefix = f"model.layers.{layer_idx}"
        orig_layer_prefix = f"model.model.layers.{layer_idx}"
        
        # Map the actual tensor names from your model to HuggingFace format
        tensor_mappings = {
            # Attention projections
            f"{orig_layer_prefix}.self_attn.q_proj.weight": f"{hf_layer_prefix}.self_attn.q_proj.weight",
            f"{orig_layer_prefix}.self_attn.k_proj.weight": f"{hf_layer_prefix}.self_attn.k_proj.weight", 
            f"{orig_layer_prefix}.self_attn.v_proj.weight": f"{hf_layer_prefix}.self_attn.v_proj.weight",
            f"{orig_layer_prefix}.self_attn.o_proj.weight": f"{hf_layer_prefix}.self_attn.o_proj.weight",
            
            # MLP projections
            f"{orig_layer_prefix}.mlp.gate_proj.weight": f"{hf_layer_prefix}.mlp.gate_proj.weight",
            f"{orig_layer_prefix}.mlp.down_proj.weight": f"{hf_layer_prefix}.mlp.down_proj.weight",
            f"{orig_layer_prefix}.mlp.up_proj.weight": f"{hf_layer_prefix}.mlp.up_proj.weight",
            
            # Layer norms
            f"{orig_layer_prefix}.input_layernorm.weight": f"{hf_layer_prefix}.input_layernorm.weight",
            f"{orig_layer_prefix}.post_attention_layernorm.weight": f"{hf_layer_prefix}.post_attention_layernorm.weight",
        }
        
        # Apply the mappings
        for orig_key, hf_key in tensor_mappings.items():
            if orig_key in original_state_dict:
                converted_state_dict[hf_key] = original_state_dict[orig_key]
            else:
                st.warning(f"Missing tensor: {orig_key}")


def add_rope_frequencies(model_config, converted_state_dict):
    """Add RoPE frequencies tensor that llama.cpp expects"""
    # Generate RoPE frequencies
    rope_freqs = generate_rope_freqs(model_config)
    
    # Add to state dict - llama.cpp expects this tensor
    converted_state_dict["rope_freqs.weight"] = rope_freqs
    st.info(f"✅ Generated rope_freqs.weight with shape {tuple(rope_freqs.shape)}")


def convert_state_dict(model):
    """Convert the model state dict to HuggingFace format"""
    original_state_dict = model.state_dict()
    converted_state_dict = {}
    model_config = extract_model_config(model)
    
    # Debug: print original tensor names
    st.write("**Original Model Tensors:**")
    tensor_count = 0
    for key in sorted(original_state_dict.keys()):
        st.write(f"{key}: {tuple(original_state_dict[key].shape)}")
        tensor_count += 1
    st.write(f"\n**Original Tensor Count = {tensor_count}**")
    
    # Convert each component
    convert_embeddings(model, original_state_dict, converted_state_dict)
    convert_normalization(original_state_dict, converted_state_dict)
    convert_output_head(original_state_dict, converted_state_dict)  # Fixed version
    convert_layer_weights(model_config, original_state_dict, converted_state_dict)
    
    # Add RoPE frequencies (this was missing!)
    add_rope_frequencies(model_config, converted_state_dict)
    
    # Debug: print converted tensor names
    st.write("\n**Converted HuggingFace Tensors:**")
    for key in sorted(converted_state_dict.keys()):
        st.write(f"{key}: {tuple(converted_state_dict[key].shape)}")
    
    st.write(f"\n**Converted Tensor Count = {len(converted_state_dict)}**")
    
    # Verify we got all the important tensors
    # 3 base tensors (embed, norm, lm_head) + 1 rope_freqs + (9 per layer)
    expected_tensor_count = 4 + (model_config.num_layers * 9)
    if len(converted_state_dict) != expected_tensor_count:
        st.warning(f"⚠️ Expected {expected_tensor_count} tensors but got {len(converted_state_dict)}")
    else:
        st.success(f"✅ Successfully converted all {len(converted_state_dict)} tensors!")
    
    return converted_state_dict


def save_hf_model(model, tokenizer, output_dir, use_f32=True):
    """Save model in HuggingFace format (fixed: saves only inner HF model)"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))
    
    # Use the inner Hugging Face model
    hf_model = model.model if hasattr(model, "model") else model
    
    # Save config
    hf_model.config.to_json_file(output_dir / "config.json")
    
    # Get proper state dict
    state_dict = hf_model.state_dict()
    processed_state_dict = {}
    for k, v in state_dict.items():
        tensor = v.contiguous()
        if not use_f32 and tensor.dtype == torch.float32:
            tensor = tensor.to(torch.float16)
        processed_state_dict[k] = tensor
    
    # Save model weights
    if use_f32:
        torch.save(processed_state_dict, output_dir / "pytorch_model.bin")
    else:
        metadata = {"format": "pt", "model_name": "llama-custom", "torch_dtype": "float16"}
        safetensors.torch.save_file(processed_state_dict, output_dir / "model.safetensors", metadata=metadata)
    
    return {
        "tensor_count": len(processed_state_dict),
        "total_size_mb": sum(p.numel() * p.element_size() for p in processed_state_dict.values()) / (1024 * 1024)
    }


def find_llamacpp_convert_script():
    """Find the llama.cpp conversion script"""
    possible_paths = [
        "convert_hf_to_gguf.py",
        "convert.py",
        "C:/Windows/System32/llama.cpp/convert_hf_to_gguf.py"
    ]
    
    if shutil.which("convert.py"):
        return "convert.py"
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def convert_to_gguf(hf_model_dir, output_path, use_f32=True):
    """Convert HuggingFace model to GGUF using llama.cpp"""
    convert_script = find_llamacpp_convert_script()
    
    if convert_script is None:
        raise RuntimeError("llama.cpp convert script not found. Please install llama.cpp and ensure convert.py is in your PATH.")
    
    cmd = [
        "python", convert_script,
        str(hf_model_dir),
        "--outfile", output_path,
        "--outtype", "f32" if use_f32 else "f16"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        raise RuntimeError(f"llama.cpp conversion failed: {result.stderr}")
    
    return {"command": " ".join(cmd), "stdout": result.stdout}


def compute_file_checksum(file_path):
    """Compute MD5 checksum of file"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def validate_export_requirements():
    """Validate that all requirements are met for export"""
    required_keys = ['training_complete', 'model', 'tokenizer']
    
    for key in required_keys:
        if key not in st.session_state or st.session_state[key] is None:
            return False
    
    return st.session_state.get('training_complete', False)


def display_export_summary(summary):
    """Display export summary"""
    st.success("✅ Export completed successfully!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("File Size", f"{summary['file_size_mb']:.1f} MB")
        st.metric("Conversion Method", "llama.cpp")
    
    with col2:
        st.metric("Tensor Count", summary['tensor_count'])
        st.metric("Data Type", "F32" if summary['use_f32'] else "F16")
    
    with st.expander("📋 Detailed Information"):
        st.write("**File Checksum:**", summary['checksum'])
        if 'command' in summary['conversion_info']:
            st.write("**Conversion Command:**", summary['conversion_info']['command'])


def main():
    """Main export interface"""
    st.title("🚀 GGUF Model Export")
    st.markdown("Export your trained model to GGUF format using llama.cpp conversion tools.")
    
    if not validate_export_requirements():
        st.info("📋 **Requirements:**")
        st.write("- Complete model training first")
        st.write("- Model and tokenizer must be loaded in session state")
        st.button("🔄 Refresh")
        return
    
    st.subheader("⚙️ Export Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.text_input("Model Name", value="llama-compatible-model")
        use_f32 = st.checkbox("Use F32 precision", value=True)
    
    with col2:
        include_debug = st.checkbox("Include debug information", value=True)
    
    if st.session_state.model is not None:
        with st.expander("📊 Model Information"):
            config = extract_model_config(st.session_state.model)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Vocabulary Size", f"{config.vocab_size:,}")
                st.metric("Hidden Size", config.hidden_size)
            with col2:
                st.metric("Number of Layers", config.num_layers)
                st.metric("Query Heads", config.num_heads)
            with col3:
                st.metric("Key/Value Heads", config.num_key_value_heads)
                st.metric("Intermediate Size", f"{config.intermediate_size:,}")
                
            # Show attention type
            if config.num_key_value_heads == config.num_heads:
                attention_type = "Multi-Head Attention (MHA)"
            elif config.num_key_value_heads == 1:
                attention_type = "Multi-Query Attention (MQA)"
            else:
                attention_type = f"Grouped Query Attention (GQA) - {config.num_heads//config.num_key_value_heads}:1 ratio"
            
            st.info(f"**Attention Type:** {attention_type}")
    
    if st.button("🚀 Export to GGUF", type="primary", use_container_width=True):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with st.spinner("Exporting model... This may take several minutes."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    hf_model_path = Path(temp_dir) / "hf_model"
                    
                    if include_debug:
                        st.info("📦 Converting to HuggingFace format...")
                    
                    hf_info = save_hf_model(st.session_state.model, st.session_state.tokenizer, hf_model_path, use_f32)
                    
                    if include_debug:
                        st.info("🔧 Converting to GGUF using llama.cpp...")
                    
                    conversion_info = convert_to_gguf(hf_model_path, tmp_path, use_f32)
                    
                    file_size = os.path.getsize(tmp_path)
                    checksum = compute_file_checksum(tmp_path)
                    
                    summary = {
                        "file_size_mb": file_size / (1024 * 1024),
                        "checksum": checksum,
                        "tensor_count": hf_info["tensor_count"],
                        "use_f32": use_f32,
                        "conversion_info": conversion_info
                    }
                    
                    display_export_summary(summary)
                    
                    with open(tmp_path, "rb") as f:
                        file_data = f.read()
                    
                    st.download_button(
                        label="📥 Download GGUF Model",
                        data=file_data,
                        file_name=f"{model_name}.gguf",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                    
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
            if include_debug:
                st.exception(e)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    main()