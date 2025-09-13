"""Common utilities for MLM framework."""

import os
import jax
import jax.numpy as jnp
import tiktoken
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def setup_jax_environment():
    """Setup JAX devices and mesh configuration."""
    if jax.default_backend() == 'tpu':
        mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ('batch', 'model'))
    else:
        num_devices = len(jax.devices())
        mesh_shape = (num_devices, 1)
        mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape), ('batch', 'model'))
    return mesh


def get_tokenizer():
    """Get the standard tokenizer with mask token."""
    tokenizer = tiktoken.get_encoding("gpt2")
    MASK_TOKEN_ID = tokenizer.n_vocab
    return tokenizer, MASK_TOKEN_ID


def setup_environment_variables():
    """Setup environment variables from various sources."""
    # Try Kaggle secrets first (for backward compatibility)
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
        os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")
        os.environ["WANDB_ENTITY"] = user_secrets.get_secret("WANDB_ENTITY")
        return True
    except ImportError:
        pass
    
    # Check if environment variables are already set
    required_vars = ["HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY"]
    if all(var in os.environ for var in required_vars):
        return True
    
    # Warn about missing environment variables
    missing_vars = [var for var in required_vars if var not in os.environ]
    print(f"Warning: Missing environment variables: {missing_vars}")
    print("Please set these variables or ensure kaggle_secrets is available.")
    return False


def create_sharding(mesh, partition_spec=P('batch', None)):
    """Create a sharding configuration for data parallelism."""
    return NamedSharding(mesh, partition_spec)


def shard_batch(batch, sharding):
    """Shard a batch across devices."""
    return {k: jax.device_put(jnp.array(v), sharding) for k, v in batch.items()}