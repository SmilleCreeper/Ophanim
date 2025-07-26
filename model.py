import torch
import torch.nn as nn
import os
import psutil
import warnings
from transformers import AutoTokenizer, LlamaForCausalLM
from rope_utils import build_rope_freqs  # if still needed for other modules
warnings.filterwarnings("ignore")

# ===============================
# Device & Memory Utility
# ===============================
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    torch.set_num_threads(min(8, os.cpu_count()))
    print("CUDA not available, using CPU")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024 / 1024
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024 / 1024
        return {
            'cpu_memory_mb': cpu_memory,
            'gpu_memory_mb': gpu_memory,
            'gpu_memory_cached_mb': gpu_memory_cached
        }
    return {'cpu_memory_mb': cpu_memory}

# ===============================
# HuggingFace-backed Transformer
# ===============================
class LlamaCompatibleTransformer(nn.Module):
    _shared_instance = None  # ✅ ensures singleton behavior

    def __new__(cls, *args, **kwargs):
        if cls._shared_instance is None:
            cls._shared_instance = super().__new__(cls)
        return cls._shared_instance

    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", **hf_kwargs):
        super().__init__()

        # ✅ Avoid reinitialization on repeated instantiation
        if hasattr(self, "_initialized") and self._initialized:
            return

        print(f"Loading HuggingFace model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            **hf_kwargs
        )

        self.model.train()  # ✅ Enable fine-tuning
        self.to(device)
        self._initialized = True

        print(f"Model loaded on {device}: {model_name}")
        print(f"Memory usage: {get_memory_usage()}")

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass (compatible with fine-tuning)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.logits if labels is None else outputs

    def generate_text(self, prompt, **gen_kwargs):
        """Convenience wrapper for inference/generation."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ===============================
# Debug Utility
# ===============================
def print_tensor_shapes(state_dict, stage):
    print(f"Tensors at {stage}:")
    for k, v in state_dict.items():
        print(f"{k}: {tuple(v.shape)}")