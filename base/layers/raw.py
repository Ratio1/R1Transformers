"""
This module contains utility functions for initializing weights and applying Rotary Position Embedding (RoPE) in transformer models.

Part of the code is adapted from The LLM Book, by Andreiy Burkov.
https://github.com/aburkov/theLMbook

"""

import math
import torch as th

from base.utils.layer_utils import rope, init_weights

class RMSNorm(th.nn.Module):
  """
  Root Mean Square Layer Normalization
  A simplified alternative to Layer Normalization that only uses RMS statistics
  """
  def __init__(self, emb_dim, epsilon=1e-8):
    super().__init__()
    self.scale = th.nn.Parameter(th.ones(emb_dim))  # Learnable scale parameter
    self.epsilon = epsilon  # Small constant for numerical stability

  def forward(self, x):
    # Compute root mean square normalization
    squared_x = x ** 2
    mean_squared = th.mean(squared_x, dim=-1, keepdim=True)
    rms = th.sqrt(mean_squared + self.epsilon)

    # Normalize and scale
    x_normalized = x / rms
    output = x_normalized * self.scale
    return output


class AttentionHead(th.nn.Module):
  """
  Single head of self-attention
  Transforms input using learned projections and computes scaled dot-product attention
  """
  def __init__(self, emb_dim, d_h):
    super().__init__()
    # Initialize projection matrices for queries, keys, and values
    self.W_Q = th.nn.Parameter(th.rand(emb_dim, d_h))
    self.W_K = th.nn.Parameter(th.rand(emb_dim, d_h))
    self.W_V = th.nn.Parameter(th.rand(emb_dim, d_h))
    self.d_h = d_h  # Dimensionality of attention head

  def forward(self, x, mask):
    # Project input into query, key, and value spaces
    Q = x @ self.W_Q
    K = x @ self.W_K
    V = x @ self.W_V

    # Apply rotary position embeddings to queries and keys
    Q, K = rope(Q), rope(K)

    # Compute attention scores with scaling factor
    scores = Q @ K.transpose(-2, -1) / th.sqrt(self.d_h)

    # Apply causal mask and attention weights
    masked_scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = th.softmax(masked_scores, dim=-1)

    return attention_weights @ V  


class MultiHeadAttention(th.nn.Module):
  """
  Multi-head attention mechanism
  Allows the model to jointly attend to information from different positions
  """
  def __init__(self, emb_dim, num_heads):
    super().__init__()
    d_h = emb_dim // num_heads  # Dimensionality of each attention head

    # Create multiple attention heads
    self.heads = th.nn.ModuleList([
        AttentionHead(emb_dim, d_h)
        for _ in range(num_heads)
    ])

    # Output projection matrix
    self.W_O = th.nn.Parameter(th.rand(emb_dim, emb_dim))
    return

  def forward(self, x, mask):
    # Process input through each attention head
    head_outputs = [head(x, mask) for head in self.heads]

    # Concatenate outputs and project to final dimensionality
    x = th.cat(head_outputs, dim=-1)
    return x @ self.W_O  
  
  
class MLP(th.nn.Module):
  """
  Multi-Layer Perceptron for transformer feed-forward network
  Uses a larger intermediate dimensionality (4x) with ReLU activation
  """
  def __init__(self, emb_dim):
    super().__init__()
    # Initialize weights and biases for two-layer feed-forward network
    self.W_1 = th.nn.Parameter(th.rand(emb_dim, emb_dim * 4))
    self.B_1 = th.nn.Parameter(th.rand(emb_dim * 4))
    self.W_2 = th.nn.Parameter(th.rand(emb_dim * 4, emb_dim))
    self.B_2 = th.nn.Parameter(th.rand(emb_dim))

  def forward(self, x):
    # First linear transformation and activation
    x = x @ self.W_1 + self.B_1
    x = th.relu(x)

    # Second linear transformation
    x = x @ self.W_2 + self.B_2
    return x


class DecoderBlock(th.nn.Module):
  """
  Single transformer decoder block
  Combines self-attention and feed-forward layers with residual connections
  """
  def __init__(self, emb_dim, num_heads):
    super().__init__()
    # Layer components
    self.norm1 = RMSNorm(emb_dim)
    self.attn = MultiHeadAttention(emb_dim, num_heads)
    self.norm2 = RMSNorm(emb_dim)
    self.mlp = MLP(emb_dim)

  def forward(self, x, mask):
    # Self-attention sub-block with residual connection
    attn_out = self.attn(self.norm1(x), mask)
    x = x + attn_out

    # Feed-forward sub-block with residual connection
    mlp_out = self.mlp(self.norm2(x))
    x = x + mlp_out
    return x
  

class DecoderLanguageModel(th.nn.Module):
  """
  Complete decoder-only transformer language model
  Processes input sequences using multiple decoder blocks and projects to vocabulary
  """
  def __init__(
    self, 
    vocab_size, 
    emb_dim=256, 
    num_heads=16, 
    num_blocks=4, 
    pad_idx=-1,
    init=True,
  ):
    super().__init__()
    # Token embedding layer
    self.embedding = th.nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

    # Stack of decoder blocks
    self.layers = th.nn.ModuleList([
        DecoderBlock(emb_dim, num_heads) for _ in range(num_blocks)
    ])

    # Output projection to vocabulary size
    self.output = th.nn.Parameter(th.rand(emb_dim, vocab_size))
    
    if init:
      init_weights(self, log_structure=True)
    return

  def forward(self, x):
    # Embed input tokens
    x = self.embedding(x)

    # Create causal attention mask
    _, seq_len, _ = x.size()
    mask = th.tril(th.ones(seq_len, seq_len, device=x.device))

    # Process through decoder blocks
    for layer in self.layers:
        x = layer(x, mask)

    # Project to vocabulary distribution
    return x @ self.output
  


if __name__ == "__main__":
  # Example usage
  vocab_size = 10000
  emb_dim = 256
  num_heads = 8
  num_blocks = 4
  pad_idx = 0

  model = DecoderLanguageModel(
    vocab_size=vocab_size, 
    emb_dim=emb_dim, 
    num_heads=num_heads, 
    num_blocks=num_blocks, 
    pad_idx=-1,
    init=True,
  )
  
  # Dummy input (batch size of 2, sequence length of 10)
  x = th.randint(0, vocab_size, (2, 10))
  
  # Forward pass
  output = model(x)
  
  print(output.shape)  # Should be (2, 10, vocab_size)