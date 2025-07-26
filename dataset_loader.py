import torch
import streamlit as st

class FastDatasetIterator:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.processed_data = []
        self._preprocess_all()
    
    def _preprocess_all(self):
        st.info("Pre-processing dataset...")
        progress_bar = st.progress(0)
        for idx, item in enumerate(self.dataset):
            pos_tokens = self.tokenizer(item['positive_example'], truncation=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
            neg_examples = []
            i = 1
            while f'negative_example_{i}' in item:
                neg_tokens = self.tokenizer(item[f'negative_example_{i}'], truncation=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
                neg_examples.append(neg_tokens)
                i += 1
            self.processed_data.append({'positive': pos_tokens, 'negatives': neg_examples})
            if idx % 10 == 0:
                progress_bar.progress((idx + 1) / len(self.dataset))
        progress_bar.empty()
        st.success(f"Pre-processed {len(self.processed_data)} examples")
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]