import torch
import streamlit as st

def project_embeddings(embed_weight: torch.Tensor, target_dim: int, st) -> torch.Tensor:
    """
    Project embeddings to match target dimension with proper preservation of semantic information.
    
    Args:
        embed_weight: Input embedding tensor of shape (vocab_size, current_dim)
        target_dim: Target embedding dimension
        st: Streamlit instance for logging
    
    Returns:
        Projected embedding tensor of shape (vocab_size, target_dim)
    """
    current_dim = embed_weight.shape[1]
    vocab_size = embed_weight.shape[0]
    
    if current_dim == target_dim:
        return embed_weight
    
    # Ensure embeddings are on CPU and in float32 for numerical stability
    embed_weight = embed_weight.to(dtype=torch.float32, device='cpu')
    
    if current_dim > target_dim:
        # Use PCA-like projection instead of simple truncation
        st.info(f"🔄 Applying learned projection: {embed_weight.shape} -> ({vocab_size}, {target_dim})")
        
        # Method 1: Use SVD for optimal dimensionality reduction
        # Center the embeddings
        embed_mean = embed_weight.mean(dim=0, keepdim=True)
        embed_centered = embed_weight - embed_mean
        
        # Compute SVD
        U, S, Vt = torch.linalg.svd(embed_centered.T, full_matrices=False)
        
        # Take top target_dim components
        projection_matrix = U[:, :target_dim]  # (current_dim, target_dim)
        
        # Project embeddings
        projected = torch.matmul(embed_centered, projection_matrix)
        
        # Add back the mean (projected to new space)
        projected_mean = torch.matmul(embed_mean, projection_matrix)
        projected = projected + projected_mean
        
        # Normalize to maintain similar scale
        original_norm = torch.norm(embed_weight, dim=1, keepdim=True).mean()
        projected_norm = torch.norm(projected, dim=1, keepdim=True).mean()
        if projected_norm > 0:
            projected = projected * (original_norm / projected_norm)
        
        return projected
        
    else:
        # Intelligent padding instead of zero padding
        st.info(f"🔄 Expanding embeddings: {embed_weight.shape} -> ({vocab_size}, {target_dim})")
        
        padding_dim = target_dim - current_dim
        
        # Method 1: Use PCA to generate meaningful padding
        if current_dim >= 2:  # Need at least 2 dimensions for PCA
            # Compute principal components of existing embeddings
            embed_mean = embed_weight.mean(dim=0, keepdim=True)
            embed_centered = embed_weight - embed_mean
            
            # Get covariance matrix
            cov_matrix = torch.matmul(embed_centered.T, embed_centered) / (vocab_size - 1)
            
            # Eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalues (descending)
            idx = torch.argsort(eigenvals, descending=True)
            eigenvecs = eigenvecs[:, idx]
            eigenvals = eigenvals[idx]
            
            # Generate padding using lower-variance directions
            # Take the least significant eigenvectors and scale by small eigenvalues
            if len(eigenvals) > padding_dim:
                padding_basis = eigenvecs[:, -padding_dim:]  # Last padding_dim eigenvectors
                padding_scales = eigenvals[-padding_dim:].sqrt() * 0.1  # Small scale
            else:
                # If we need more padding than available eigenvectors, use random orthogonal vectors
                padding_basis = torch.randn(current_dim, padding_dim)
                # Orthogonalize against existing eigenvectors
                for i in range(min(current_dim, padding_dim)):
                    for j in range(min(len(eigenvecs[0]), current_dim)):
                        padding_basis[:, i] -= torch.dot(padding_basis[:, i], eigenvecs[:, j]) * eigenvecs[:, j]
                    padding_basis[:, i] /= torch.norm(padding_basis[:, i])
                padding_scales = torch.full((padding_dim,), 0.01)  # Very small scale
            
            # Generate padding values
            padding_values = torch.matmul(embed_centered, padding_basis) * padding_scales.unsqueeze(0)
            
        else:
            # Fallback: small random padding
            padding_values = torch.randn(vocab_size, padding_dim) * 0.01
        
        # Concatenate original embeddings with intelligent padding
        expanded = torch.cat([embed_weight, padding_values], dim=1)
        
        return expanded