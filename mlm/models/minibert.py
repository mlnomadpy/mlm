"""MiniBERT model implementation."""

import flax.nnx as nnx
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from mlm.core.models import BaseMLMModel, ModelRegistry


class TransformerBlock(nnx.Module):
    """A standard Transformer block with bidirectional self-attention."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, *, rngs: nnx.Rngs, rate: float = 0.1, mesh=None):
        if mesh is not None:
            kernel_init = nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model')))
        else:
            kernel_init = nnx.initializers.xavier_uniform()

        self.mha = nnx.MultiHeadAttention(num_heads=num_heads, in_features=embed_dim, use_bias=False, kernel_init=kernel_init, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=rate, rngs=rngs)
        self.ffn = nnx.Sequential(
            nnx.Linear(embed_dim, ff_dim, kernel_init=kernel_init, rngs=rngs),
            nnx.gelu,
            nnx.Linear(ff_dim, embed_dim, kernel_init=kernel_init, rngs=rngs)
        )
        self.dropout2 = nnx.Dropout(rate=rate, rngs=rngs)
        self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(self, inputs, training: bool = False):
        x = self.norm1(inputs)
        attention_output = self.mha(inputs_q=x, decode=False)
        attention_output = self.dropout1(attention_output, deterministic=not training)
        out1 = inputs + attention_output
        x = self.norm2(out1)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, deterministic=not training)
        return out1 + ffn_output


class TokenAndPositionEmbedding(nnx.Module):
    """Combines token and positional embeddings."""
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(num_embeddings=maxlen, features=embed_dim, rngs=rngs)

    def __call__(self, x):
        positions = jnp.arange(0, x.shape[1])[None, :]
        return self.token_emb(x) + self.pos_emb(positions)


@ModelRegistry.register("minibert")
class MiniBERT(nnx.Module, BaseMLMModel):
    """A compact BERT-like model for masked language modeling."""
    
    def __init__(self, vocab_size: int, maxlen: int, embed_dim: int, num_heads: int, 
                 ff_dim: int, num_layers: int, *, rngs: nnx.Rngs, rate: float = 0.1, mesh=None):
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.rate = rate
        
        self.embedding = TokenAndPositionEmbedding(
            maxlen=maxlen, vocab_size=vocab_size, embed_dim=embed_dim, rngs=rngs
        )
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, rngs=rngs, rate=rate, mesh=mesh) 
            for _ in range(num_layers)
        ]
        self.dropout = nnx.Dropout(rate=rate, rngs=rngs)
        
        if mesh is not None:
            kernel_init = nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P('model', None)))
        else:
            kernel_init = nnx.initializers.xavier_uniform()
        
        self.mlm_head = nnx.Linear(embed_dim, vocab_size, kernel_init=kernel_init, rngs=rngs)

    def __call__(self, x, training: bool = False):
        # Embedding layer
        x = self.embedding(x)
        x = self.dropout(x, deterministic=not training)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        # MLM prediction head
        return self.mlm_head(x)
    
    def get_config(self):
        """Get model configuration."""
        return {
            "model_type": "minibert",
            "vocab_size": self.vocab_size,
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_layers": self.num_layers,
            "rate": self.rate,
        }