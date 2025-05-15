"""
This module contains utility functions for initializing weights and applying Rotary Position Embedding (RoPE) in transformer models.

Part of the code is adapted from The LLM Book, by Andreiy Burkov.
https://github.com/aburkov/theLMbook

"""

import torch as th


def rope(x, theta_base=10000.0):
  """
  Implements Rotary Position Embedding (RoPE) for transformer attention.
  RoPE encodes position information through rotation matrices applied to pairs of dimensions.

  Args:
      x: Input tensor of shape (batch_size, seq_len, emb_dim)
      theta_base: Base for computing rotation frequencies (default: 10000.0)

  Returns:
      Tensor with position information encoded through rotations
  """
  batch_size, seq_len, emb_dim = x.size()
  assert emb_dim % 2 == 0, "Embedding dimensionality must be even for RoPE"

  # Generate sequence position indices
  pos = th.arange(0, seq_len, dtype=th.float32, device=x.device)
  pos = pos.unsqueeze(0).expand(batch_size, seq_len)

  # Compute frequency bands for each dimension pair
  # Modified: frequencies start from p=1 and use (p-1) in exponent
  p = th.arange(1, emb_dim // 2 + 1, dtype=th.float32, device=x.device)
  theta_p = 1.0 / (theta_base ** (2 * (p - 1) / emb_dim))

  # Compute rotation angles for each position and frequency
  pos = pos.unsqueeze(-1)
  theta = pos * theta_p

  # Compute rotation components
  sin_theta = th.sin(theta)
  cos_theta = th.cos(theta)

  # Split input into alternating dimensions
  x1 = x[..., 0::2]  # Dimensions at indices 0,2,4,...
  x2 = x[..., 1::2]  # Dimensions at indices 1,3,5,...

  # Apply 2D rotations to each pair
  x_rotated_1 = x1 * cos_theta - x2 * sin_theta
  x_rotated_2 = x1 * sin_theta + x2 * cos_theta

  # Recombine rotated pairs into final output
  x_rotated = th.stack((x_rotated_1, x_rotated_2), dim=-1).reshape(batch_size, seq_len, emb_dim)

  return x_rotated


def init_weights(model):
  """
  Initialize the weights of different model components using appropriate schemes.
  Each layer type receives specialized initialization for optimal training.
  """
  for module in model.modules():
    if isinstance(module, th.nn.Linear):
      # Xavier uniform initialization for linear layers
      # Helps maintain variance across network layers
      th.nn.init.xavier_uniform_(module.weight)
      if module.bias is not None:
        th.nn.init.zeros_(module.bias)  # Initialize biases to zero
    elif isinstance(module, th.nn.Embedding):
      # Initialize embedding layers with normal distribution
      th.nn.init.normal_(module.weight, mean=0, std=0.02)
      if module.padding_idx is not None:
        # Ensure padding tokens have zero embeddings
        with th.no_grad():
          module.weight[module.padding_idx].fill_(0)
    elif module.__class__.__name__ == "AttentionHead":
      # Initialize query, key, and value projection matrices
      # Xavier uniform helps maintain good gradient flow
      th.nn.init.xavier_uniform_(module.W_Q)
      th.nn.init.xavier_uniform_(module.W_K)
      th.nn.init.xavier_uniform_(module.W_V)
    elif  module.__class__.__name__ == "MultiHeadAttention":
      # Initialize output projection matrix for attention mechanism
      th.nn.init.xavier_uniform_(module.W_O)
    elif  module.__class__.__name__ == "DecoderLanguageModel":
      # Initialize final output projection layer
      th.nn.init.xavier_uniform_(module.output)
    elif  module.__class__.__name__ == "RMSNorm":
      # Initialize RMSNorm scale parameters to ones
      # This starts with identity transformation
      th.nn.init.ones_(module.scale)
    elif  module.__class__.__name__ == "MLP":
      # Initialize feed-forward network parameters
      th.nn.init.xavier_uniform_(module.W_1)
      th.nn.init.xavier_uniform_(module.W_2)
      th.nn.init.zeros_(module.B_1)
      th.nn.init.zeros_(module.B_2)
