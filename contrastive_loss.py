import torch
import torch.nn as nn
import math

def optimized_contrastive_loss(model, positive_ids, negative_ids_list, tokenizer, base_weight=0.01):
    """
    Optimized contrastive loss function for training language models.
    
    Args:
        model: The language model to compute loss for
        positive_ids: Tensor of positive example token IDs
        negative_ids_list: List of tensors containing negative example token IDs
        tokenizer: Tokenizer used for padding token ID
        base_weight: Base weight for contrastive loss component
    
    Returns:
        tuple: (total_loss, positive_loss, negative_loss)
    """
    # Get device from model
    device = next(model.parameters()).device
    
    # Move tensors to device
    positive_ids = positive_ids.to(device)
    negative_ids_list = [neg_ids.to(device) for neg_ids in negative_ids_list]
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    num_negatives = len(negative_ids_list)
    
    if num_negatives == 0:
        pos_logits = model(positive_ids.unsqueeze(0))
        targets = positive_ids[1:].contiguous()
        logits_for_loss = pos_logits[0, :-1, :].contiguous()
        return nn.CrossEntropyLoss(ignore_index=pad_token_id)(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            targets.view(-1)
        ), 0.0, 0.0
    
    contrastive_weight = base_weight * math.exp(-0.1 * num_negatives)
    pos_logits = model(positive_ids.unsqueeze(0))
    pos_targets = positive_ids[1:].contiguous()
    pos_logits_for_loss = pos_logits[0, :-1, :].contiguous()
    
    pos_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)(
        pos_logits_for_loss.view(-1, pos_logits_for_loss.size(-1)),
        pos_targets.view(-1)
    )
    
    max_len = max(neg.size(0) for neg in negative_ids_list)
    batch_negatives = []
    valid_lengths = []
    
    for neg_ids in negative_ids_list:
        if neg_ids.size(0) < max_len:
            padding = torch.full((max_len - neg_ids.size(0),), pad_token_id, dtype=neg_ids.dtype, device=device)
            padded_neg = torch.cat([neg_ids, padding])
        else:
            padded_neg = neg_ids
        batch_negatives.append(padded_neg)
        valid_lengths.append(neg_ids.size(0))
    
    if batch_negatives:
        batch_tensor = torch.stack(batch_negatives)
        neg_logits = model(batch_tensor)
        neg_losses = []
        for j, (neg_ids, valid_len) in enumerate(zip(negative_ids_list, valid_lengths)):
            if valid_len > 1:
                neg_targets = neg_ids[1:valid_len].contiguous()
                neg_logits_for_loss = neg_logits[j, :valid_len-1, :].contiguous()
                neg_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)(
                    neg_logits_for_loss.view(-1, neg_logits_for_loss.size(-1)),
                    neg_targets.view(-1)
                )
                neg_losses.append(neg_loss)
        if neg_losses:
            avg_neg_loss = torch.stack(neg_losses).mean()
            total_loss = pos_loss - contrastive_weight * avg_neg_loss
            return total_loss, pos_loss.item(), avg_neg_loss.item()
    
    return pos_loss, pos_loss.item(), 0.0