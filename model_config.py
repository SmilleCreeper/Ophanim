from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    def create_hf_config(self, tokenizer, use_f32: bool = True) -> Dict[str, Any]:
        """Create HuggingFace-compatible config for Llama architecture."""
        
        # Определяем правильные token IDs из токенизатора
        bos_token_id = getattr(tokenizer, 'bos_token_id', 128000)
        eos_token_id = getattr(tokenizer, 'eos_token_id', 128001)
        pad_token_id = getattr(tokenizer, 'pad_token_id', eos_token_id)

        return {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_heads,  # Важно для Llama
            "hidden_act": "silu",
            "max_position_embeddings": self.max_position_embeddings,
            "initializer_range": 0.02,
            "rms_norm_eps": self.rms_norm_eps,
            "use_cache": True,
            "rope_theta": self.rope_theta,
            "rope_scaling": None,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "mlp_bias": False,
            "tie_word_embeddings": True,  # ВАЖНО: указываем что веса связаны!
            "torch_dtype": "float32" if use_f32 else "float16",
            "transformers_version": "4.36.0",
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id
        }