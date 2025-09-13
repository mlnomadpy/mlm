"""BERT model implementation using Flax NNX."""

from typing import Optional
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


class TransformerBlock(nnx.Module):
    """A standard Transformer block with bidirectional self-attention."""
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ff_dim: int, 
        *, 
        rngs: nnx.Rngs, 
        rate: float = 0.1,
        mesh: Optional[Mesh] = None
    ):
        """Initialize TransformerBlock.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            rngs: Random number generators
            rate: Dropout rate
            mesh: JAX mesh for partitioning (optional)
        """
        if mesh is not None:
            kernel_init = nnx.with_partitioning(
                nnx.initializers.xavier_uniform(), 
                NamedSharding(mesh, P(None, 'model'))
            )
        else:
            kernel_init = nnx.initializers.xavier_uniform()

        self.mha = nnx.MultiHeadAttention(
            num_heads=num_heads, 
            in_features=embed_dim, 
            use_bias=False, 
            kernel_init=kernel_init, 
            rngs=rngs
        )
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
        """Forward pass through the transformer block."""
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
        """Initialize embeddings.
        
        Args:
            maxlen: Maximum sequence length
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            rngs: Random number generators
        """
        self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(num_embeddings=maxlen, features=embed_dim, rngs=rngs)

    def __call__(self, x):
        """Apply token and position embeddings."""
        positions = jnp.arange(0, x.shape[1])[None, :]
        return self.token_emb(x) + self.pos_emb(positions)


class MiniBERT(nnx.Module):
    """A miniBERT transformer model for Masked Language Modeling."""
    
    def __init__(
        self, 
        maxlen: int, 
        vocab_size: int, 
        embed_dim: int, 
        num_heads: int, 
        feed_forward_dim: int, 
        num_transformer_blocks: int, 
        rngs: nnx.Rngs,
        mesh: Optional[Mesh] = None
    ):
        """Initialize MiniBERT model.
        
        Args:
            maxlen: Maximum sequence length
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            feed_forward_dim: Feed-forward dimension
            num_transformer_blocks: Number of transformer blocks
            rngs: Random number generators
            mesh: JAX mesh for partitioning (optional)
        """
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, rngs=rngs)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, feed_forward_dim, rngs=rngs, mesh=mesh) 
            for _ in range(num_transformer_blocks)
        ]
        
        if mesh is not None:
            kernel_init = nnx.with_partitioning(
                nnx.initializers.xavier_uniform(), 
                NamedSharding(mesh, P(None, 'model'))
            )
            bias_init = nnx.with_partitioning(
                nnx.initializers.zeros_init(), 
                NamedSharding(mesh, P('model'))
            )
        else:
            kernel_init = nnx.initializers.xavier_uniform()
            bias_init = nnx.initializers.zeros_init()
            
        self.output_layer = nnx.Linear(
            in_features=embed_dim, 
            out_features=vocab_size,
            kernel_init=kernel_init,
            bias_init=bias_init,
            use_bias=False, 
            rngs=rngs
        )

    def __call__(self, inputs, training: bool = False):
        """Forward pass through the model."""
        x = self.embedding_layer(inputs)
        for block in self.transformer_blocks:
            x = block(x, training=training)
        return self.output_layer(x)

    def embed(self, inputs, training: bool = False):
        """Gets embeddings before the final output layer for MTEB."""
        x = self.embedding_layer(inputs)
        for block in self.transformer_blocks:
            x = block(x, training=training)
        return x