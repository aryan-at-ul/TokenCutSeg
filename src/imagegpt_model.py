#src/imagegpt_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


##复制  https://github.com/karpathy/nanoGPT


# -----------------------------------------------------------------------------
# Helper: Truncated Normal Initialization
# -----------------------------------------------------------------------------
def truncated_normal_(tensor, mean=0.0, std=0.02, a=-0.04, b=0.04):
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    return tensor

# -----------------------------------------------------------------------------
# Embed Module: Token + Position Embeddings
# -----------------------------------------------------------------------------
class Embed(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_position_embeddings: int,
        dropout: float = 0.1,
        hidden_size: int = None,
        initializer_range: float = 0.02,
    ):
        """
        Args:
            vocab_size: Number of tokens (e.g. codebook size).
            embed_dim: Dimensionality of token and position embeddings.
            max_position_embeddings: Maximum sequence length.
            dropout: Dropout probability.
            hidden_size: If given and different from embed_dim, project embeddings.
            initializer_range: Std for weight initialization.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim)
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.hidden_mapping = None
        if hidden_size is not None and hidden_size != embed_dim:
            self.hidden_mapping = nn.Linear(embed_dim, hidden_size)
        truncated_normal_(self.word_embeddings.weight, std=initializer_range)
        truncated_normal_(self.position_embeddings.weight, std=initializer_range)
        if self.hidden_mapping is not None:
            truncated_normal_(self.hidden_mapping.weight, std=initializer_range)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        # Create position ids: 0, 1, ..., seq_length-1
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)
        x = word_emb + pos_emb
        x = self.layernorm(x)
        x = self.dropout(x)
        if self.hidden_mapping is not None:
            x = self.hidden_mapping(x)
        return x

# -----------------------------------------------------------------------------
# Attention Module: Multihead Self-Attention with Residual & LayerNorm
# -----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        hidden_dropout: float,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(hidden_dropout)
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
        truncated_normal_(self.multihead_attn.in_proj_weight, std=initializer_range)
        truncated_normal_(self.multihead_attn.out_proj.weight, std=initializer_range)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask)
        attn_output = self.dropout(attn_output)
        output = self.layernorm(attn_output + residual)
        return output

# -----------------------------------------------------------------------------
# MLP Module: Feedforward Network with Residual & LayerNorm
# -----------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout: float,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
        truncated_normal_(self.fc1.weight, std=initializer_range)
        truncated_normal_(self.fc2.weight, std=initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.layernorm(x + residual)
        return x

# -----------------------------------------------------------------------------
# Transformer Layer: One Block (Attention + MLP)
# -----------------------------------------------------------------------------
class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        hidden_dropout: float,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.attention = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            initializer_range=initializer_range,
        )
        self.mlp = Mlp(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout=hidden_dropout,
            initializer_range=initializer_range,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.attention(x, attn_mask=attn_mask)
        x = self.mlp(x)
        return x

# -----------------------------------------------------------------------------
# MLM Head: Projects Hidden States to Vocabulary Logits
# -----------------------------------------------------------------------------
class MlmLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        truncated_normal_(self.dense.weight, std=initializer_range)

    def forward(self, hidden_states: torch.Tensor, embedding_weights: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layernorm(x)
        logits = torch.matmul(x, embedding_weights.t()) + self.bias
        return logits

# -----------------------------------------------------------------------------
# Transformer: Stacks of Layers + Embedding + MLM Head
# -----------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 4,
        intermediate_size: int = 1024,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 3136,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.embed = Embed(
            vocab_size=vocab_size,
            embed_dim=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=hidden_dropout_prob,
            hidden_size=hidden_size,
            initializer_range=initializer_range,
        )
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_probs_dropout_prob,
                hidden_dropout=hidden_dropout_prob,
                initializer_range=initializer_range,
            )
            for _ in range(num_hidden_layers)
        ])
        self.mlm_layer = MlmLayer(hidden_size, vocab_size, initializer_range=initializer_range)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embed(input_ids)  # (B, T, hidden_size)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        # When not returning features, project to vocabulary logits.
        embedding_weights = self.embed.word_embeddings.weight
        logits = self.mlm_layer(x, embedding_weights)
        return logits

# -----------------------------------------------------------------------------
# ImageGPT: Wrapping the Transformer with an Option to Return Latent Features
# -----------------------------------------------------------------------------
class ImageGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,  # Maximum sequence length (e.g., flattened spatial tokens)
        n_layer: int = 12,   # Number of transformer layers
        n_head: int = 4,     # Number of attention heads
        n_embd: int = 256,   # Hidden size (embedding dimension)
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
    ):
        """
        Args:
            vocab_size: Codebook size.
            block_size: Maximum sequence length.
            n_layer: Number of transformer layers.
            n_head: Number of attention heads.
            n_embd: Hidden size (and embedding dimension). For segmentation, set this to 256.
            hidden_dropout_prob: Dropout rate for hidden layers.
            attention_probs_dropout_prob: Dropout rate for attention.
            initializer_range: Std for weight initialization.
        """
        super().__init__()
        intermediate_size = 4 * n_embd  # By convention, MLP hidden size is 4x n_embd.
        self.transformer = Transformer(
            vocab_size=vocab_size,
            hidden_size=n_embd,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=block_size,
            initializer_range=initializer_range,
        )

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor = None, return_features: bool = True) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (B, T) with token indices.
            attn_mask: Optional attention mask.
            return_features: If True (default), return latent features of shape [B, T, n_embd];
                             if False, return vocabulary logits.
        """
        x = self.transformer.embed(input_ids)
        for layer in self.transformer.layers:
            x = layer(x, attn_mask=attn_mask)
        if return_features:
            return x  # Latent features: expected shape [B, T, n_embd] (e.g., [1, 3136, 256])
        else:
            embedding_weights = self.transformer.embed.word_embeddings.weight
            logits = self.transformer.mlm_layer(x, embedding_weights)
            return logits

# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example configuration:
    #   - Codebook size (vocab_size) is 1024.
    #   - For a 224x224 image downsampled by a factor of 4, the spatial resolution is 56x56 = 3136.
    vocab_size = 1024
    batch_size = 1
    block_size = 3136  # (image_size // 4)**2
    # GPT configuration: 12 layers, 4 heads, and embedding dimension 256.
    model = ImageGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=12,
        n_head=4,
        n_embd=256,  # This ensures latent features are [B, T, 256]
    )
    # Create dummy input tokens (e.g., output of a VQGAN encoder)
    dummy_input = torch.randint(0, vocab_size, (batch_size, block_size))
    # Get latent features (for segmentation, for example)
    features = model(dummy_input)  # With default return_features=True
    print("Features shape:", features.shape)  # Expected: [batch_size, block_size, 256]