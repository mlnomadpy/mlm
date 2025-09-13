"""Data processing utilities for MLM framework."""

import random
import jax.numpy as jnp
from datasets import load_dataset


def get_fineweb_data(config):
    """Load and prepare Fineweb dataset for training."""
    dataset = load_dataset(
        "HuggingFaceFW/fineweb", 
        name="sample-100BT", 
        split="train", 
        streaming=True
    )
    return dataset


def tokenize_and_mask(example, tokenizer, maxlen=512, mask_prob=0.15):
    """Tokenize text and apply masking for MLM training."""
    text = example["text"]
    
    # Tokenize text
    tokens = tokenizer.encode(text)
    
    # Truncate or pad to maxlen
    if len(tokens) > maxlen:
        tokens = tokens[:maxlen]
    else:
        tokens = tokens + [tokenizer.eot_token] * (maxlen - len(tokens))
    
    # Create labels (copy of tokens)
    labels = tokens.copy()
    
    # Apply masking
    for i in range(len(tokens)):
        if random.random() < mask_prob:
            rand_val = random.random()
            if rand_val < 0.8:  # 80% mask
                tokens[i] = tokenizer.n_vocab  # MASK_TOKEN_ID
            elif rand_val < 0.9:  # 10% random token
                tokens[i] = random.randint(0, tokenizer.n_vocab - 1)
            # 10% keep original (else clause not needed)
    
    return {
        "input_ids": tokens,
        "labels": labels
    }


def prepare_batch_iterator(dataset, tokenizer, batch_size, maxlen=512, mask_prob=0.15):
    """Create a batch iterator from the dataset."""
    def tokenize_fn(example):
        return tokenize_and_mask(example, tokenizer, maxlen, mask_prob)
    
    # Apply tokenization
    tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
    
    return tokenized_dataset