import torch
import json
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional
import safetensors.torch
from projecting import project_embeddings
from convert_embeddings import convert_embeddings
from convert_normalization import convert_normalization
from convert_output_head import convert_output_head
from convert_layer_weights import convert_layer_weights
from convert_log_debug_info import convert_log_debug_info
from calculate_parameters import calculate_total_parameters
from model_config import ModelConfig


class HuggingFaceConverter:
    """Converts custom models to HuggingFace format compatible with llama.cpp."""

    def __init__(self, model, tokenizer, use_f32: bool = True, include_debug_info: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.use_f32 = use_f32
        self.include_debug_info = include_debug_info
        self.model_config = self._extract_model_config()

    def _extract_model_config(self) -> ModelConfig:
        """Extract model configuration from the loaded model."""
        return ModelConfig(
            vocab_size=self.model.embed_tokens.num_embeddings,
            hidden_size=self.model.hidden_size,
            num_layers=len(self.model.layers),
            num_heads=self.model.layers[0].attention.num_heads,
            intermediate_size=self.model.intermediate_size,
            # Важно: добавляем параметры RoPE
            rope_theta=self.model.layers[0].attention.rope.base if hasattr(self.model.layers[0].attention.rope,
                                                                           'base') else 10000.0,
            max_position_embeddings=self.model.layers[0].attention.rope.max_position_embeddings if hasattr(
                self.model.layers[0].attention.rope, 'max_position_embeddings') else 2048
        )

    def calculate_total_parameters(self) -> int:
        """Calculate the total number of parameters in the model."""
        return calculate_total_parameters(self.model_config)

    def convert_to_huggingface_format(self, output_dir: Path) -> Dict[str, Any]:
        """Convert model to HuggingFace format and save to directory."""
        output_dir.mkdir(exist_ok=True)

        # Save tokenizer
        st.write("💾 Saving tokenizer...")
        self.tokenizer.save_pretrained(str(output_dir))

        # Create config.json compatible with Llama
        config_dict = self.model_config.create_hf_config(self.tokenizer, self.use_f32)
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        if self.include_debug_info:
            st.write("✅ Config saved:", config_dict)

        # Save model weights
        st.write("💾 Saving model weights...")
        model_files = self._save_model_weights(output_dir)

        return {
            "config": config_dict,
            "model_files": model_files,
            "tokenizer_files": list(output_dir.glob("tokenizer*")) + list(output_dir.glob("special_tokens*"))
        }

    def _save_model_weights(self, output_dir: Path) -> Dict[str, Any]:
        """Save model weights in format compatible with llama.cpp."""

        # Convert state dict to HuggingFace Llama naming convention
        converted_state_dict = self._convert_state_dict()

        # ВАЖНО: Проверяем и исправляем связанные веса
        if "model.embed_tokens.weight" in converted_state_dict and "lm_head.weight" not in converted_state_dict:
            # Если lm_head отсутствует, но веса должны быть связаны
            if self.include_debug_info:
                st.write("⚠️ lm_head.weight отсутствует, копируем из embed_tokens (tied weights)")
            converted_state_dict["lm_head.weight"] = converted_state_dict["model.embed_tokens.weight"].clone()

        # Убеждаемся, что все тензоры contiguous и правильного типа
        processed_state_dict = {}
        for k, v in converted_state_dict.items():
            tensor = v.contiguous()
            if not self.use_f32 and tensor.dtype == torch.float32:
                tensor = tensor.to(torch.float16)
            processed_state_dict[k] = tensor

        # Choose save format
        if self.use_f32:
            # Save as PyTorch checkpoint
            model_file = output_dir / "pytorch_model.bin"
            torch.save(processed_state_dict, model_file)
            files_saved = ["pytorch_model.bin"]
        else:
            # Save as SafeTensors for better compatibility
            model_file = output_dir / "model.safetensors"
            metadata = {
                "format": "pt",
                "model_name": "llama-custom",
                "torch_dtype": "float16"
            }
            safetensors.torch.save_file(processed_state_dict, model_file, metadata=metadata)
            files_saved = ["model.safetensors"]

        # Create model index if needed
        total_size = sum(param.numel() * param.element_size() for param in processed_state_dict.values())

        if len(files_saved) == 1:
            # Single file model
            weight_map = {key: files_saved[0] for key in processed_state_dict.keys()}
        else:
            # Multi-file model (for very large models)
            weight_map = {}  # Would need implementation for sharding

        model_index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map
        }

        with open(output_dir / "pytorch_model.bin.index.json", "w") as f:
            json.dump(model_index, f, indent=2)

        return {
            "files": files_saved,
            "total_size_mb": total_size / (1024 * 1024),
            "tensor_count": len(processed_state_dict)
        }

    def _convert_state_dict(self) -> Dict[str, torch.Tensor]:
        """Convert custom model state dict to HuggingFace Llama format."""
        original_state_dict = self.model.state_dict()
        converted_state_dict = {}

        # Convert different components
        convert_embeddings(self.model, original_state_dict, converted_state_dict, self.include_debug_info, st)
        convert_normalization(original_state_dict, converted_state_dict)
        convert_output_head(original_state_dict, converted_state_dict, st)
        convert_layer_weights(self.model_config, original_state_dict, converted_state_dict, self.include_debug_info, st)

        # Log debug information if enabled
        if self.include_debug_info:
            convert_log_debug_info(original_state_dict, converted_state_dict, st)

        if hasattr(self.model, 'rope_freqs_weight'):
            converted_state_dict['rope_freqs.weight'] = self.model.rope_freqs_weight

        # Debug print: show all tensor names and shapes after conversion
        if self.include_debug_info:
            print("[DEBUG] Tensors after PyTorch→HF conversion:")
            for k, v in converted_state_dict.items():
                print(f"{k}: {tuple(v.shape)}")
            print(f"[DEBUG] Total tensors: {len(converted_state_dict)}")

        return converted_state_dict