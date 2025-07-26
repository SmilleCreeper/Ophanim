# training_loop.py

import torch
import time
import streamlit as st
from contrastive_loss import optimized_contrastive_loss
from performance_enchancer import perform_garbage_collection, cuda_memory_stats
from model import get_memory_usage, device


def process_training_step(model, item, tokenizer, optimizer, config, step_count):
    loss, pos_loss, neg_loss = optimized_contrastive_loss(
        model, item['positive'], item['negatives'], tokenizer
    )
    
    loss = loss / config['batch_accumulation']
    loss.backward()
    step_count += 1
    
    if step_count % config['batch_accumulation'] == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    return loss.item() * config['batch_accumulation'], pos_loss, neg_loss, step_count


def handle_remaining_gradients(model, optimizer, config, step_count):
    if step_count % config['batch_accumulation'] != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()


def perform_garbage_collection_if_needed(epoch, config, ui_components):
    if (epoch + 1) % config['gc_frequency'] == 0:
        pre_gc_memory = get_memory_usage()
        collected_objects = perform_garbage_collection()
        post_gc_memory = get_memory_usage()
        
        if device.type == 'cuda':
            gpu_stats = cuda_memory_stats()
            ui_components['memory_info'].info(f"🗑️ GC (Epoch {epoch + 1}): "
                                            f"Collected {collected_objects} objects | "
                                            f"CPU: {pre_gc_memory['cpu_memory_mb']:.1f}MB → {post_gc_memory['cpu_memory_mb']:.1f}MB | "
                                            f"GPU: {gpu_stats['allocated']:.1f}GB used, {gpu_stats['reserved']:.1f}GB reserved")
        else:
            memory_saved = pre_gc_memory['cpu_memory_mb'] - post_gc_memory['cpu_memory_mb']
            ui_components['memory_info'].info(f"🗑️ GC (Epoch {epoch + 1}): "
                                            f"Collected {collected_objects} objects, "
                                            f"Memory: {pre_gc_memory['cpu_memory_mb']:.1f}MB → {post_gc_memory['cpu_memory_mb']:.1f}MB "
                                            f"({'freed' if memory_saved > 0 else 'no change'}: {abs(memory_saved):.1f}MB)")


def update_training_status(epoch, config, epoch_time, avg_loss, avg_pos_loss, avg_neg_loss, optimizer, ui_components):
    current_lr = optimizer.param_groups[0]['lr']
    current_memory = get_memory_usage()
    
    if device.type == 'cuda':
        gpu_stats = cuda_memory_stats()
        ui_components['status_text'].text(
            f"Epoch {epoch + 1}/{config['num_epochs']} | "
            f"Time: {epoch_time:.4f}s | "
            f"Loss: {avg_loss:.4f} | "
            f"Pos: {avg_pos_loss:.4f} | "
            f"Neg: {avg_neg_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"GPU: {gpu_stats['allocated']:.1f}GB"
        )
    else:
        ui_components['status_text'].text(
            f"Epoch {epoch + 1}/{config['num_epochs']} | "
            f"Time: {epoch_time:.4f}s | "
            f"Loss: {avg_loss:.4f} | "
            f"Pos: {avg_pos_loss:.4f} | "
            f"Neg: {avg_neg_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Memory: {current_memory['cpu_memory_mb']:.1f}MB"
        )


def update_loss_chart(epoch, config, losses, pos_losses, neg_losses, ui_components):
    if epoch % 1 == 0 or epoch == config['num_epochs'] - 1:
        with ui_components['loss_chart_placeholder'].container():
            st.line_chart({
                'Total Loss': losses,
                'Positive Loss': pos_losses,
                'Negative Loss': neg_losses
            })


def run_training_loop(model, optimizer, config, ui_components, valid_indices):
    losses = []
    pos_losses = []
    neg_losses = []
    
    total_samples = len(valid_indices)
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        model.train()
        
        epoch_loss = 0
        epoch_pos_loss = 0
        epoch_neg_loss = 0
        step_count = 0
        
        ui_components['main_progress_bar'].progress(epoch / config['num_epochs'])
        
        for batch_idx, idx in enumerate(valid_indices):
            item = st.session_state.dataset_iterator[idx]
            
            loss, pos_loss, neg_loss, step_count = process_training_step(
                model, item, st.session_state.tokenizer, optimizer, config, step_count
            )
            
            epoch_loss += loss
            epoch_pos_loss += pos_loss
            epoch_neg_loss += neg_loss
            
            ui_components['epoch_progress_bar'].progress((batch_idx + 1) / total_samples)
            
            if config['sync_cuda'] and device.type == 'cuda':
                torch.cuda.synchronize()
        
        handle_remaining_gradients(model, optimizer, config, step_count)
        
        num_valid = len(valid_indices)
        avg_loss = float(epoch_loss / num_valid)
        avg_pos_loss = float(epoch_pos_loss / num_valid)
        avg_neg_loss = float(epoch_neg_loss / num_valid)
        
        perform_garbage_collection_if_needed(epoch, config, ui_components)
        
        epoch_time = time.time() - epoch_start_time
        losses.append(avg_loss)
        pos_losses.append(avg_pos_loss)
        neg_losses.append(avg_neg_loss)
        
        update_training_status(epoch, config, epoch_time, avg_loss, avg_pos_loss, avg_neg_loss, optimizer, ui_components)
        update_loss_chart(epoch, config, losses, pos_losses, neg_losses, ui_components)
    
    return losses, pos_losses, neg_losses