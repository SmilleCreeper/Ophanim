import streamlit as st
import os
import torch
import json
from model import get_memory_usage
from dataset_loader import FastDatasetIterator

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'dataset_iterator' not in st.session_state:
    st.session_state.dataset_iterator = None

st.set_page_config(page_title="Llama-Compatible RoPE Transformer", layout="wide")
st.title("Llama-Compatible RoPE Transformer")
st.caption("Enhanced with Rotary Position Embedding (RoPE) and Llama architecture")

# Home Section: System Information
st.header("System Information")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Check Memory"):
        memory_mb = get_memory_usage()
        st.metric("RAM Usage", f"{memory_mb:.1f} MB")
with col2:
    cpu_count = os.cpu_count()
    st.metric("CPU Cores", cpu_count)
with col3:
    threads = torch.get_num_threads()
    st.metric("PyTorch Threads", threads)

# Dataset Upload Section
st.header("Dataset Upload")
uploaded_file = st.file_uploader("Upload JSON dataset", type=['json'])

if uploaded_file is not None:
    try:
        dataset = json.load(uploaded_file)
        st.session_state.dataset = dataset
        st.success(f"Dataset loaded: {len(dataset)} examples")
        
        with st.expander("Dataset Analysis"):
            if dataset:
                sample = dataset[0]
                neg_count = len([k for k in sample.keys() if k.startswith('negative_example_')])
                pos_len = len(sample['positive_example'])
                neg_lens = [len(sample.get(f'negative_example_{i}', '')) for i in range(1, neg_count + 1)]
                avg_neg_len = sum(neg_lens) / len(neg_lens) if neg_lens else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Examples", len(dataset))
                with col2:
                    st.metric("Avg Negatives", neg_count)
                with col3:
                    st.metric("Avg Pos Length", f"{pos_len:.0f} chars")
                with col4:
                    st.metric("Avg Neg Length", f"{avg_neg_len:.0f} chars")
                st.json(sample)
        
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
else:
    st.info("Please upload a JSON dataset to begin training")