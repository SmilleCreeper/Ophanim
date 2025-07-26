# training_preload.py

import streamlit as st
import torch
import gc
from transformers import AutoTokenizer, AutoModel
from model import device


def load_tokenizer():
    """Load and configure tokenizer"""
    with st.spinner("Loading tokenizer..."):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        st.session_state.tokenizer = tokenizer
        return tokenizer


def load_pretrained_embeddings(use_mixed_precision):
    """Load pretrained embeddings from the model"""
    pretrained_embeddings = None
    with st.spinner("Loading embeddings..."):
        # Load model with appropriate device mapping
        if device.type == 'cuda':
            pretrained_model = AutoModel.from_pretrained(
                "meta-llama/Llama-3.2-3B-Instruct",
                torch_dtype=torch.float16 if use_mixed_precision else torch.float32,
                device_map="auto"
            )
        else:
            pretrained_model = AutoModel.from_pretrained(
                "meta-llama/Llama-3.2-3B-Instruct",
                torch_dtype=torch.float32,
                device_map=None
            )
        
        pretrained_embeddings = pretrained_model.embed_tokens.weight.data.clone().detach()
        # Convert to float32 for training stability
        if pretrained_embeddings.dtype != torch.float32:
            pretrained_embeddings = pretrained_embeddings.float()
        vocab_size = pretrained_embeddings.shape[0]
        del pretrained_model
        
        # Clear GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return pretrained_embeddings, vocab_size