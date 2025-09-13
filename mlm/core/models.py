"""Model registry and base classes for MLM framework."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable
import flax.nnx as nnx


class BaseMLMModel(ABC):
    """Abstract base class for MLM models."""
    
    @abstractmethod
    def __call__(self, x, training: bool = False):
        """Forward pass of the model."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        pass


class MTEBModel:
    """Wrapper for MTEB evaluation."""
    
    def __init__(self, model, tokenizer, maxlen=512):
        self.model = model
        self.tokenizer = tokenizer
        self.maxlen = maxlen
    
    def encode(self, sentences, batch_size=32, **kwargs):
        """Encode sentences for MTEB evaluation."""
        import jax.numpy as jnp
        
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            batch_tokens = []
            
            for sentence in batch_sentences:
                tokens = self.tokenizer.encode(sentence)
                if len(tokens) > self.maxlen:
                    tokens = tokens[:self.maxlen]
                else:
                    tokens = tokens + [self.tokenizer.eot_token] * (self.maxlen - len(tokens))
                batch_tokens.append(tokens)
            
            batch_tokens = jnp.array(batch_tokens)
            embeddings = self.model(batch_tokens, training=False)
            
            # Mean pooling
            embeddings = jnp.mean(embeddings, axis=1)
            all_embeddings.append(embeddings)
        
        return jnp.concatenate(all_embeddings, axis=0)


class ModelRegistry:
    """Registry for available models."""
    
    _models: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a model."""
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, name: str):
        """Get a model by name."""
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found. Available models: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def list_models(cls):
        """List all registered models."""
        return list(cls._models.keys())