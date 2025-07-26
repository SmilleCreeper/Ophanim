# pages/training.py

import streamlit as st
import torch
import torch.optim as optim
import time
import gc
from model import (
    LlamaCompatibleTransformer, 
    get_memory_usage, 
    device
)
from performance_enchancer import perform_garbage_collection, cuda_memory_stats
from dataset_loader import FastDatasetIterator
from training_loop import run_training_loop
from training_preload import load_tokenizer

def display_device_info():
    if device.type == 'cuda':
        st.sidebar.success(f"🚀 Using CUDA: {torch.cuda.get_device_name()}")
        memory_stats = cuda_memory_stats()
        if memory_stats:
            st.sidebar.info(f"GPU Memory: {memory_stats['allocated']:.1f}GB / {memory_stats['reserved']:.1f}GB")
    else:
        st.sidebar.info("💻 Using CPU")

def optimize_cuda_settings():
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()

def setup_ui_configuration():
    st.sidebar.header("Configuration")
    display_device_info()
    
    st.sidebar.subheader("Training Settings")
    num_epochs = st.sidebar.number_input("Epochs", value=10, min_value=5, max_value=1000)
    learning_rate = st.sidebar.selectbox("Learning Rate", [0.0001, 0.0005, 0.001, 0.002], index=2)
    batch_accumulation = st.sidebar.number_input("Gradient Accumulation", value=4, min_value=1, max_value=8)
    
    st.sidebar.subheader("Memory Management")
    gc_frequency = st.sidebar.number_input("Garbage Collection Every N Epochs", value=10, min_value=1, max_value=20)
    
    if device.type == 'cuda':
        sync_cuda = st.sidebar.checkbox("Sync CUDA", value=False)
    else:
        sync_cuda = False
    
    return {
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_accumulation': batch_accumulation,
        'gc_frequency': gc_frequency,
        'sync_cuda': sync_cuda
    }

def create_model_and_optimizer(config):
    model = LlamaCompatibleTransformer()
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    return model, optimizer

def setup_training_ui():
    st.subheader("Training Progress")
    main_progress_bar = st.progress(0)
    epoch_progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart_placeholder = st.empty()
    memory_info = st.empty()
    
    if device.type == 'cuda':
        initial_gpu_stats = cuda_memory_stats()
        st.info(f"Initial GPU Memory: {initial_gpu_stats['allocated']:.1f}GB allocated, {initial_gpu_stats['reserved']:.1f}GB reserved")
    
    return {
        'main_progress_bar': main_progress_bar,
        'epoch_progress_bar': epoch_progress_bar,
        'status_text': status_text,
        'loss_chart_placeholder': loss_chart_placeholder,
        'memory_info': memory_info
    }

def prepare_training_data():
    indices = torch.randperm(len(st.session_state.dataset_iterator))
    valid_indices = []
    for idx in indices:
        item = st.session_state.dataset_iterator[idx.item()]
        if len(item['negatives']) > 0:
            valid_indices.append(idx.item())
    return valid_indices

def display_training_results(start_time, initial_memory):
    final_collected = perform_garbage_collection()
    total_time = time.time() - start_time
    final_memory = get_memory_usage()
    
    if device.type == 'cuda':
        final_gpu_stats = cuda_memory_stats()
        st.success(f"✅ Training completed! "
                  f"Time: {total_time:.1f}s | "
                  f"CPU: {initial_memory['cpu_memory_mb']:.1f}MB → {final_memory['cpu_memory_mb']:.1f}MB | "
                  f"GPU: {final_gpu_stats['allocated']:.1f}GB used | "
                  f"Final cleanup: {final_collected} objects")
    else:
        st.success(f"✅ Training completed! "
                  f"Time: {total_time:.1f}s | "
                  f"Memory: {initial_memory['cpu_memory_mb']:.1f}MB → {final_memory['cpu_memory_mb']:.1f}MB | "
                  f"Final cleanup: {final_collected} objects")

def main():
    st.header("Training")
    
    config = setup_ui_configuration()
    
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset on the Dataset page before training")
        return
    
    if st.button("Start Training", type="primary"):
        try:
            optimize_cuda_settings()
            start_total_time = time.time()
            initial_memory = get_memory_usage()
            
            tokenizer = load_tokenizer()
            model, optimizer = create_model_and_optimizer(config)
            
            st.session_state.model = model
            st.session_state.dataset_iterator = FastDatasetIterator(st.session_state.dataset, tokenizer)
            
            ui_components = setup_training_ui()
            
            valid_indices = prepare_training_data()
            
            losses, pos_losses, neg_losses = run_training_loop(
                model, optimizer, config, ui_components, valid_indices
            )
            
            ui_components['main_progress_bar'].progress(1.0)
            ui_components['epoch_progress_bar'].progress(1.0)
            
            display_training_results(start_total_time, initial_memory)
            
            st.session_state.training_complete = True
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()