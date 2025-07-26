import torch
import gc


def perform_garbage_collection():
    """Perform garbage collection and clear CUDA cache if available"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    collected = gc.collect()
    return collected


def setup_mixed_precision_training():
    """Setup mixed precision training for CUDA"""
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        return scaler
    return None


def cuda_memory_stats():
    """Get detailed CUDA memory statistics"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
            'max_reserved': torch.cuda.max_memory_reserved() / 1024**3,    # GB
        }
    return None