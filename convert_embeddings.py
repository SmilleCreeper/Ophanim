import torch
import streamlit as st
from typing import Dict
from projecting import project_embeddings

def convert_embeddings(model, original_state_dict: Dict[str, torch.Tensor], converted_state_dict: Dict[str, torch.Tensor], include_debug_info: bool, st) -> None:
    """Handle embedding tokens and projection conversion."""
    if "embed_tokens.weight" in original_state_dict:
        embed_weight = original_state_dict["embed_tokens.weight"]
        
        # Check if we need to project embeddings to match output dimension
        if "lm_head.weight" in original_state_dict:
            output_weight = original_state_dict["lm_head.weight"]
            output_dim = output_weight.shape[1] if len(output_weight.shape) > 1 else output_weight.shape[0]
            embed_dim = embed_weight.shape[1]
            
            if embed_dim != output_dim:
                st.info(f"🔄 Dimension mismatch detected: embed={embed_dim}, output={output_dim}")
                
                # Check if there's an explicit projection layer
                if hasattr(model, 'embedding_projection') and not isinstance(model.embedding_projection, torch.nn.Identity):
                    if "embedding_projection.weight" in original_state_dict:
                        proj_weight = original_state_dict["embedding_projection.weight"]
                        
                        # Handle different projection matrix orientations
                        if proj_weight.shape[0] == embed_dim and proj_weight.shape[1] == output_dim:
                            # Standard orientation: (embed_dim, output_dim)
                            embed_weight = torch.matmul(embed_weight, proj_weight)
                            if include_debug_info:
                                st.write(f"✅ Applied explicit projection: {proj_weight.shape}")
                        elif proj_weight.shape[1] == embed_dim and proj_weight.shape[0] == output_dim:
                            # Transposed orientation: (output_dim, embed_dim) - need to transpose
                            proj_weight_t = proj_weight.T  # Now (embed_dim, output_dim)
                            embed_weight = torch.matmul(embed_weight, proj_weight_t)
                            if include_debug_info:
                                st.write(f"✅ Applied transposed projection: {proj_weight.shape} -> {proj_weight_t.shape}")
                        else:
                            st.warning(f"⚠️ Projection shape mismatch: {proj_weight.shape}, expected ({embed_dim}, {output_dim}) or ({output_dim}, {embed_dim})")
                            embed_weight = project_embeddings(embed_weight, output_dim, st)
                    else:
                        st.warning("⚠️ Model has embedding_projection but no weights found")
                        embed_weight = project_embeddings(embed_weight, output_dim, st)
                else:
                    # Use improved projection method
                    embed_weight = project_embeddings(embed_weight, output_dim, st)
                    
                # Verify final dimensions
                if embed_weight.shape[1] != output_dim:
                    st.error(f"❌ Projection failed: got {embed_weight.shape[1]}, expected {output_dim}")
                    raise ValueError(f"Embedding projection failed: {embed_weight.shape[1]} != {output_dim}")
        
        converted_state_dict["model.embed_tokens.weight"] = embed_weight

    # Handle embedding projection if it exists (and wasn't already applied above)
    if "embedding_projection.weight" in original_state_dict:
        # Only save as separate parameter if we didn't already apply it
        if "model.embed_tokens.weight" not in converted_state_dict or embed_dim == output_dim:
            projection_weight = original_state_dict["embedding_projection.weight"]
            if include_debug_info:
                st.write("💾 Saving embedding projection as separate parameter")
            converted_state_dict["model.embed_projection.weight"] = projection_weight