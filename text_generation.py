import torch
import warnings
warnings.filterwarnings("ignore")

def memory_efficient_generate(model, tokenizer, prompt, max_length=50, temperature=0.8, device=None):
    """
    Memory-efficient text generation with automatic device detection and mixed precision support.
    
    Args:
        model: The transformer model for generation
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Input text prompt
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature for randomness
        device: Device to use (auto-detected if None)
    
    Returns:
        Generated text string
    """
    # Auto-detect device if not provided
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    input_ids = tokenizer(prompt, return_tensors='pt', truncation=False)['input_ids'].to(device)
    generated = input_ids.clone()
    
    with torch.no_grad():
        for step in range(max_length):
            # Sliding window to prevent memory overflow
            if generated.size(1) > 1024:
                generated = generated[:, -512:]
            
            # Use mixed precision if available on CUDA
            if device.type == 'cuda' and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    logits = model(generated)
            else:
                logits = model(generated)
                
            # Temperature sampling with top-k filtering
            next_token_logits = logits[0, -1, :] / temperature
            top_k = 10
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            next_token_probs = torch.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(next_token_probs, 1)
            next_token = top_k_indices[next_token_idx].unsqueeze(0)
            
            # Stop generation on EOS or PAD tokens
            if next_token.item() in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                break
                
            generated = torch.cat([generated, next_token], dim=1)
    
    return tokenizer.decode(generated[0], skip_special_tokens=False)

def batch_generate(model, tokenizer, prompts, max_length=50, temperature=0.8, device=None):
    """
    Generate text for multiple prompts in batch for efficiency.
    
    Args:
        model: The transformer model for generation
        tokenizer: Tokenizer for encoding/decoding text
        prompts: List of input text prompts
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature for randomness
        device: Device to use (auto-detected if None)
    
    Returns:
        List of generated text strings
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    results = []
    
    # Process prompts individually to avoid memory issues
    for prompt in prompts:
        result = memory_efficient_generate(
            model, tokenizer, prompt, max_length, temperature, device
        )
        results.append(result)
    
    return results

def generate_with_beam_search(model, tokenizer, prompt, num_beams=3, max_length=50, device=None):
    """
    Simple beam search implementation for better quality generation.
    
    Args:
        model: The transformer model for generation
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Input text prompt
        num_beams: Number of beams for beam search
        max_length: Maximum number of tokens to generate
        device: Device to use (auto-detected if None)
    
    Returns:
        Generated text string (best beam)
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    input_ids = tokenizer(prompt, return_tensors='pt', truncation=False)['input_ids'].to(device)
    
    # Initialize beams: (sequence, score)
    beams = [(input_ids.clone(), 0.0)]
    
    with torch.no_grad():
        for step in range(max_length):
            candidates = []
            
            for sequence, score in beams:
                # Sliding window for memory efficiency
                if sequence.size(1) > 1024:
                    sequence = sequence[:, -512:]
                
                # Get predictions
                if device.type == 'cuda' and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        logits = model(sequence)
                else:
                    logits = model(sequence)
                
                next_token_logits = logits[0, -1, :]
                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                # Get top candidates
                top_k = min(num_beams * 2, log_probs.size(0))
                top_log_probs, top_indices = torch.topk(log_probs, top_k)
                
                for i in range(top_k):
                    token_id = top_indices[i].unsqueeze(0).unsqueeze(0)
                    new_sequence = torch.cat([sequence, token_id], dim=1)
                    new_score = score + top_log_probs[i].item()
                    candidates.append((new_sequence, new_score))
            
            # Select best beams
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:num_beams]
            
            # Check if all beams ended
            all_ended = True
            for sequence, _ in beams:
                last_token = sequence[0, -1].item()
                if last_token not in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                    all_ended = False
                    break
            
            if all_ended:
                break
    
    # Return best beam
    best_sequence, _ = beams[0]
    return tokenizer.decode(best_sequence[0], skip_special_tokens=True)