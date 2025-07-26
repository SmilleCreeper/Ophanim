from model_config import ModelConfig

def calculate_total_parameters(config: ModelConfig) -> int:
    """Calculate the total number of parameters in the model."""
    # Embedding parameters
    embed_params = config.vocab_size * config.hidden_size
    
    # Attention parameters (Q, K, V, O projections)
    attention_params = config.num_layers * (4 * config.hidden_size * config.hidden_size)
    
    # Feed-forward parameters (gate, up, down)
    ffn_params = config.num_layers * (config.hidden_size * config.intermediate_size * 3)
    
    # Normalization parameters (attention norm + ffn norm per layer + output norm)
    norm_params = config.num_layers * (2 * config.hidden_size) + config.hidden_size
    
    # Output head parameters
    output_params = config.vocab_size * config.hidden_size
    
    return embed_params + attention_params + ffn_params + norm_params + output_params