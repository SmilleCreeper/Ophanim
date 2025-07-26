import streamlit as st
import os
import tempfile
import hashlib
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pytorch_to_hf import HuggingFaceConverter
from model_config import ModelConfig
from rope_utils import build_rope_freqs

@dataclass
class ExportSettings:
    model_name: str
    use_f32: bool
    include_debug_info: bool = True
    llamacpp_path: Optional[str] = None

class LlamaCppExporter:
    """Main class for handling GGUF model exports using llama.cpp."""
    
    def __init__(self, model, tokenizer, settings: ExportSettings):
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        self.hf_converter = HuggingFaceConverter(
            model, 
            tokenizer, 
            use_f32=settings.use_f32,
            include_debug_info=settings.include_debug_info
        )
    
    def _add_rope_freqs_tensor(self, gguf_tensors: Dict[str, Any], config: ModelConfig, hf_model_path: Path = None):
        """
        Ensure rope_freqs.weight tensor is present in GGUF tensors as a 1D tensor of shape (rope_dim,),
        where rope_dim = head_dim // 2, for LLaMA/llama.cpp compatibility. Adds robust error handling and type checks.
        """
        import torch
        rope_freqs = None
        try:
            head_dim = int(config.hidden_size) // int(config.num_heads)
            rope_dim = head_dim // 2
            if rope_dim <= 0:
                raise ValueError(f"Invalid rope_dim: {rope_dim}. Check hidden_size ({config.hidden_size}) and num_heads ({config.num_heads})")
        except Exception as e:
            st.error(f"Error computing head_dim/rope_dim: {e}")
            raise

        # Try to load from HuggingFace model files if available
        if hf_model_path is not None:
            pt_file = hf_model_path / "pytorch_model.bin"
            if pt_file.exists():
                try:
                    state_dict = torch.load(pt_file, map_location="cpu")
                    if "rope_freqs.weight" in state_dict:
                        rope_freqs = state_dict["rope_freqs.weight"]
                        if not isinstance(rope_freqs, torch.Tensor):
                            raise TypeError("rope_freqs.weight in checkpoint is not a torch.Tensor!")
                        if rope_freqs.ndim > 1:
                            rope_freqs = rope_freqs.reshape(-1)
                        rope_freqs = rope_freqs.contiguous()
                        if self.settings.include_debug_info:
                            st.write(f"Loaded rope_freqs.weight from HF checkpoint with shape: {tuple(rope_freqs.shape)}")
                except Exception as e:
                    st.error(f"Error loading rope_freqs.weight from checkpoint: {e}")
                    rope_freqs = None

        # If not found, build it
        if rope_freqs is None:
            try:
                rope_freqs = build_rope_freqs(head_dim, config.max_position_embeddings)
                if not isinstance(rope_freqs, torch.Tensor):
                    raise TypeError("build_rope_freqs did not return a torch.Tensor!")
                rope_freqs = rope_freqs.contiguous()
                if self.settings.include_debug_info:
                    st.write(f"Built rope_freqs.weight with shape: {tuple(rope_freqs.shape)}")
            except Exception as e:
                st.error(f"Error building rope_freqs.weight: {e}")
                raise

        # Always slice to [:rope_dim] for llama.cpp compatibility
        try:
            if rope_freqs.shape[0] < rope_dim:
                raise ValueError(f"rope_freqs tensor too short: shape {rope_freqs.shape}, expected at least {rope_dim}")
            rope_freqs = rope_freqs[:rope_dim]
            # Remove all singleton dimensions and flatten to 1D
            rope_freqs = rope_freqs.reshape(-1).contiguous()
            if rope_freqs.dtype != torch.float32:
                rope_freqs = rope_freqs.float()
            if self.settings.include_debug_info:
                st.write(f"Exporting rope_freqs.weight with final shape: {tuple(rope_freqs.shape)} (should be ({rope_dim},)), dtype: {rope_freqs.dtype}")
            gguf_tensors['rope_freqs.weight'] = rope_freqs
        except Exception as e:
            st.error(f"Error finalizing rope_freqs tensor for export: {e}")
            raise

    def export_to_gguf(self, output_path: str) -> Dict[str, Any]:
        """Main export function that uses llama.cpp for conversion."""
        try:
            # Create temporary directory for intermediate files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Step 1: Export model in HuggingFace format
                hf_model_path = temp_path / "hf_model"
                hf_model_path.mkdir(exist_ok=True)
                
                st.info("📦 Preparing model files for llama.cpp...")
                model_files = self.hf_converter.convert_to_huggingface_format(hf_model_path)

                # Step 1.5: Prepare GGUF tensors and add rope_freqs.weight
                gguf_tensors = {}
                config = self.hf_converter.model_config
                self._add_rope_freqs_tensor(gguf_tensors, config, hf_model_path)

                # Step 2: Convert using llama.cpp
                st.info("🔧 Converting to GGUF using llama.cpp...")
                conversion_info = self._convert_with_llamacpp(hf_model_path, output_path)

                # Optionally, inject gguf_tensors into the GGUF file if fallback is used
                # (If using llama.cpp, it should handle rope_freqs.weight, but fallback should add it manually)
                # If you implement fallback GGUF writing, make sure to use gguf_tensors

                # Step 3: Generate export summary
                return self._create_export_summary(output_path, model_files, conversion_info)
                
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
            raise
    
    def _find_llamacpp_convert_script(self) -> Optional[str]:
        """Find llama.cpp convert script."""
        possible_paths = [
            self.settings.llamacpp_path,
            "C:/Windows/System32/llama.cpp/convert_hf_to_gguf.py"
        ]
        
        # Also check if convert.py is in PATH
        if shutil.which("convert.py"):
            possible_paths.append("convert.py")
        
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
        
        return None
    
    def _convert_with_llamacpp(self, model_dir: Path, output_path: str) -> Dict[str, Any]:
        """Convert model using llama.cpp convert script."""
        
        # Find convert script
        convert_script = self._find_llamacpp_convert_script()
        
        if convert_script is None:
            # Fallback to manual GGUF creation
            st.warning("⚠️ llama.cpp convert script not found. Using fallback method.")
            return self._fallback_gguf_creation(model_dir, output_path)
        
        # Prepare conversion command
        cmd = [
            "python", convert_script,
            str(model_dir),
            "--outfile", output_path,
            "--outtype", "f32" if self.settings.use_f32 else "f16"
        ]
        
        if self.settings.include_debug_info:
            st.write(f"🔧 Running command: {' '.join(cmd)}")
        
        try:
            # Run conversion
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                st.error(f"❌ Conversion failed with return code {result.returncode}")
                st.error(f"stderr: {result.stderr}")
                st.error(f"stdout: {result.stdout}")
                raise RuntimeError(f"llama.cpp conversion failed: {result.stderr}")
            
            if self.settings.include_debug_info:
                st.success("✅ llama.cpp conversion completed")
                if result.stdout:
                    st.write("Conversion output:", result.stdout)
            
            return {
                "method": "llama.cpp",
                "command": " ".join(cmd),
                "stdout": result.stdout,
                "conversion_time": "completed"
            }
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Conversion timed out after 5 minutes")
        except Exception as e:
            st.error(f"❌ Error running llama.cpp conversion: {str(e)}")
            raise
    
    def _fallback_gguf_creation(self, model_dir: Path, output_path: str) -> Dict[str, Any]:
        """Fallback method when llama.cpp is not available."""
        st.info("🔄 Using fallback GGUF creation method...")
        
        # This would require implementing a basic GGUF writer
        # For now, we'll raise an error with instructions
        error_msg = """
        llama.cpp convert script not found.
        """
        
        st.error(error_msg)
        raise RuntimeError("llama.cpp not available and no fallback implemented")
    
    def _create_export_summary(self, output_path: str, model_files: Dict, 
                             conversion_info: Dict) -> Dict[str, Any]:
        """Create a summary of the export process."""
        file_size = os.path.getsize(output_path)
        checksum = self._compute_file_checksum(output_path)
        
        return {
            "file_path": output_path,
            "file_size_mb": file_size / (1024 * 1024),
            "checksum": checksum,
            "model_config": self.hf_converter.model_config,
            "total_parameters": self.hf_converter.calculate_total_parameters(),
            "model_files": model_files,
            "conversion_info": conversion_info,
            "settings": self.settings
        }
    
    def _compute_file_checksum(self, file_path: str) -> str:
        """Compute MD5 checksum of the exported file."""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()