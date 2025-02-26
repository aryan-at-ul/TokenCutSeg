# src/imagegpt_model2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 复制 from tamming transformer

# -----------------------------------------------------------------------------
# Helper: Truncated Normal Initialization
# -----------------------------------------------------------------------------
def truncated_normal_(tensor, mean=0.0, std=0.02, a=-0.04, b=0.04):
    """
    Fills the input tensor with values drawn from a truncated normal distribution.
    Args:
        tensor: an n-dimensional torch.Tensor
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    return tensor

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        hidden_dropout: float,
        initializer_range: float = 0.02,
        use_flash_attn: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        assert self.head_dim * num_attention_heads == hidden_size, "hidden_size must be divisible by num_attention_heads"
        
        # Combined projection for Q, K, V to save parameters
        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.resid_dropout = nn.Dropout(hidden_dropout)
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
        
        # Use flash attention when available for better performance
        self.use_flash_attn = use_flash_attn and hasattr(F, 'scaled_dot_product_attention')
        
        # Initialize weights
        truncated_normal_(self.c_attn.weight, std=initializer_range)
        truncated_normal_(self.c_proj.weight, std=initializer_range)
        
        # Register buffer for causal mask to avoid recomputation
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(1024, 1024, dtype=torch.bool)).view(1, 1, 1024, 1024)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, return_attention: bool = False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        residual = x
        
        # Apply layer norm first (Pre-LN Transformer architecture)
        x = self.layernorm(x)
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.hidden_size, dim=2)
        
        # Reshape: (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        k = k.view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Attention map for potential return
        att = None
        att_soft = None
        
        # Efficient attention using Flash Attention when available
        if self.use_flash_attn and T <= 1024 and not return_attention:
            with torch.cuda.amp.autocast(enabled=False):
                # Flash attention expects (B, nh, T, hs)
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=True
                )
        else:
            # Manual implementation of attention
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply causal mask - the mask is pre-registered as a buffer
            if T <= 1024:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            else:
                # For longer sequences, create a dynamic causal mask
                causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                att = att.masked_fill(~causal_mask.view(1, 1, T, T), float('-inf'))
            
            # Apply additional attention mask if provided
            if attn_mask is not None:
                # attn_mask: (B, T) -> (B, 1, 1, T)
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
                att = att.masked_fill(attn_mask == 0, float('-inf'))
            
            att_soft = F.softmax(att, dim=-1)
            att_dropped = self.attn_dropout(att_soft)
            
            y = att_dropped @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        
        # Add residual connection
        output = y + residual
        
        if return_attention and att_soft is not None:
            # Return attention map along with output
            return output, att_soft
        
        return output
    
    def get_attention_map(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        """Extract attention maps without computing the full forward pass"""
        B, T, C = x.size()
        
        # Apply layer norm
        x = self.layernorm(x)
        
        # Calculate query, key for attention map
        q, k, _ = self.c_attn(x).split(self.hidden_size, dim=2)
        
        # Reshape: (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        k = k.view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if T <= 1024:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        else:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~causal_mask.view(1, 1, T, T), float('-inf'))
        
        # Apply additional attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            att = att.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        att = F.softmax(att, dim=-1)
        
        return att

class Mlp(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout: float,
        initializer_range: float = 0.02,
        use_gated_mlp: bool = False
    ):
        super().__init__()
        self.use_gated_mlp = use_gated_mlp
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
        
        if use_gated_mlp:
            # Gated MLP (similar to GLU) for better performance
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.act_fn = nn.GELU()
            
            # Initialize weights
            truncated_normal_(self.gate_proj.weight, std=initializer_range)
            truncated_normal_(self.up_proj.weight, std=initializer_range)
            truncated_normal_(self.down_proj.weight, std=initializer_range)
        else:
            # Standard MLP
            self.fc1 = nn.Linear(hidden_size, intermediate_size)
            self.activation = nn.GELU()
            self.fc2 = nn.Linear(intermediate_size, hidden_size)
            
            # Initialize weights
            truncated_normal_(self.fc1.weight, std=initializer_range)
            truncated_normal_(self.fc2.weight, std=initializer_range)
            
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layernorm(x)
        
        if self.use_gated_mlp:
            # Gated MLP forward pass (improved performance)
            gate = self.act_fn(self.gate_proj(x))
            up = self.up_proj(x)
            x = gate * up
            x = self.down_proj(x)
        else:
            # Standard MLP forward pass
            x = self.fc1(x)
            x = self.activation(x)
            x = self.fc2(x)
            
        x = self.dropout(x)
        return x + residual    


class MlmLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True
    ):
        super().__init__()
        self.tie_word_embeddings = tie_word_embeddings
        
        if not tie_word_embeddings:
            # If not tying weights, use a projection layer
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.activation = nn.GELU()
            self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
            self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
            self.bias = nn.Parameter(torch.zeros(vocab_size))
            
            # Initialize weights
            truncated_normal_(self.dense.weight, std=initializer_range)
            truncated_normal_(self.decoder.weight, std=initializer_range)
        else:
            # If tying weights, we only need bias
            self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states: torch.Tensor, embedding_weights: torch.Tensor = None) -> torch.Tensor:
        if self.tie_word_embeddings:
            if embedding_weights is None:
                raise ValueError("embedding_weights must be provided when tie_word_embeddings=True")
            # Simply project using the transposed embedding matrix
            logits = hidden_states @ embedding_weights.T + self.bias
        else:
            # Use the full projection pipeline
            x = self.dense(hidden_states)
            x = self.activation(x)
            x = self.layernorm(x)
            logits = self.decoder(x) + self.bias
            
        return logits



class Embed(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_position_embeddings: int,
        dropout: float = 0.1,
        hidden_size: int = None,
        initializer_range: float = 0.02,
        position_embedding_type: str = 'absolute',
    ):
        """
        Args:
            vocab_size: Number of tokens (e.g. codebook size).
            embed_dim: Dimensionality of token and position embeddings.
            max_position_embeddings: Maximum sequence length.
            dropout: Dropout probability.
            hidden_size: If given and different from embed_dim, project embeddings.
            initializer_range: Std for weight initialization.
            position_embedding_type: 'absolute' or 'relative'
        """
        super().__init__()
        self.position_embedding_type = position_embedding_type
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.max_position_embeddings = max_position_embeddings
        
        if position_embedding_type == 'absolute':
            self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim)
        else:
            # 2D Learnable relative position bias similar to Swin Transformer
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * 64 - 1) * (2 * 64 - 1), embed_dim)
            )
            self.register_buffer("relative_position_index", self._get_2d_relative_position_index(64, 64))
            
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.hidden_mapping = None
        if hidden_size is not None and hidden_size != embed_dim:
            self.hidden_mapping = nn.Linear(embed_dim, hidden_size)
            
        # Initialize weights
        truncated_normal_(self.word_embeddings.weight, std=initializer_range)
        if position_embedding_type == 'absolute':
            truncated_normal_(self.position_embeddings.weight, std=initializer_range)
        else:
            truncated_normal_(self.relative_position_bias_table, std=initializer_range)
            
        if self.hidden_mapping is not None:
            truncated_normal_(self.hidden_mapping.weight, std=initializer_range)

    def _get_2d_relative_position_index(self, h, w):
        # Get 2D relative position index for a grid of size (h, w)
        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += h - 1
        relative_coords[:, :, 1] += w - 1
        relative_coords[:, :, 0] *= 2 * w - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.size()
        
        # Apply token embeddings
        word_emb = self.word_embeddings(input_ids)
        
        # Apply position embeddings based on type
        if self.position_embedding_type == 'absolute':
            # Create position ids: 0, 1, ..., seq_length-1
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
            pos_emb = self.position_embeddings(position_ids)
            x = word_emb + pos_emb
        else:
            # Get relative position embedding
            if seq_length > self.max_position_embeddings:
                # For long sequences, fall back to standard absolute embeddings
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = torch.clamp(position_ids, 0, self.max_position_embeddings - 1)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
                pos_emb = self.position_embeddings(position_ids)
                x = word_emb + pos_emb
            else:
                x = word_emb
                # Relative position embeddings are applied in the attention layer
        
        x = self.layernorm(x)
        x = self.dropout(x)
        
        if self.hidden_mapping is not None:
            x = self.hidden_mapping(x)
            
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        hidden_dropout: float,
        initializer_range: float = 0.02,
        use_gated_mlp: bool = False,
        use_flash_attn: bool = False
    ):
        super().__init__()
        self.attention = CausalSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            initializer_range=initializer_range,
            use_flash_attn=use_flash_attn
        )
        self.mlp = Mlp(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout=hidden_dropout,
            initializer_range=initializer_range,
            use_gated_mlp=use_gated_mlp
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, return_attention: bool = False):
        if return_attention:
            attn_output, attn_map = self.attention(x, attn_mask=attn_mask, return_attention=True)
            mlp_output = self.mlp(attn_output)
            return mlp_output, attn_map
        else:
            attn_output = self.attention(x, attn_mask=attn_mask)
            mlp_output = self.mlp(attn_output)
            return mlp_output



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
        tie_word_embeddings: bool = True,
        use_gated_mlp: bool = False,
        position_embedding_type: str = 'absolute',
        use_flash_attn: bool = False,
        output_attentions: bool = False
    ):
        super().__init__()
        self.embedding_dim = hidden_size
        self.vocab_size = vocab_size
        self.use_gated_mlp = use_gated_mlp
        self.tie_word_embeddings = tie_word_embeddings
        self.position_embedding_type = position_embedding_type
        self.output_attentions = output_attentions
        
        self.embed = Embed(
            vocab_size=vocab_size,
            embed_dim=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=hidden_dropout_prob,
            hidden_size=hidden_size,
            initializer_range=initializer_range,
            position_embedding_type=position_embedding_type
        )
        
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_probs_dropout_prob,
                hidden_dropout=hidden_dropout_prob,
                initializer_range=initializer_range,
                use_gated_mlp=use_gated_mlp,
                use_flash_attn=use_flash_attn
            )
            for _ in range(num_hidden_layers)
        ])
        
        self.ln_f = nn.LayerNorm(hidden_size, eps=1e-12)
        
        self.mlm_layer = MlmLayer(
            hidden_size=hidden_size, 
            vocab_size=vocab_size, 
            initializer_range=initializer_range,
            tie_word_embeddings=tie_word_embeddings
        )

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor = None, return_attention: bool = False):
        x = self.embed(input_ids)  # (B, T, hidden_size)
        
        attention_outputs = [] if return_attention or self.output_attentions else None
        
        for layer in self.layers:
            if return_attention or self.output_attentions:
                x, attn_map = layer(x, attn_mask=attn_mask, return_attention=True)
                attention_outputs.append(attn_map)
            else:
                x = layer(x, attn_mask=attn_mask)
            
        x = self.ln_f(x)
        
        # When not returning features, project to vocabulary logits
        embedding_weights = self.embed.word_embeddings.weight if self.tie_word_embeddings else None
        logits = self.mlm_layer(x, embedding_weights)
        
        if return_attention or self.output_attentions:
            return logits, x, attention_outputs
        else:
            return logits, x


# Updated ImageGPT class with attention output support
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        tie_word_embeddings: bool = True,
        use_gated_mlp: bool = False,
        position_embedding_type: str = 'absolute',
        use_flash_attn: bool = False,
        mask_token_id: int = None,
        output_attentions: bool = False
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
            tie_word_embeddings: Whether to tie input and output embeddings.
            use_gated_mlp: Whether to use Gated MLP for better performance.
            position_embedding_type: Type of position embeddings ('absolute' or 'relative')
            use_flash_attn: Whether to use flash attention for faster computation.
            mask_token_id: Optional explicit mask token ID.
            output_attentions: Whether to output attention maps from all layers.
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
            tie_word_embeddings=tie_word_embeddings,
            use_gated_mlp=use_gated_mlp,
            position_embedding_type=position_embedding_type,
            use_flash_attn=use_flash_attn,
            output_attentions=output_attentions
        )
        
        # Register mask token ID if provided
        if mask_token_id is not None:
            self.register_buffer('mask_token_id', torch.tensor(mask_token_id, dtype=torch.long))
            
        # Flag for attention output
        self.output_attentions = output_attentions

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor = None, 
                return_features: bool = False, return_attn: bool = False):
        """
        Args:
            input_ids: Tensor of shape (B, T) with token indices.
            attn_mask: Optional attention mask.
            return_features: If True, return latent features of shape [B, T, n_embd].
            return_attn: If True, return attention maps from all layers.
        """
        if return_attn or self.output_attentions:
            logits, features, attn_outputs = self.transformer(
                input_ids, 
                attn_mask=attn_mask,
                return_attention=True
            )
        else:
            logits, features = self.transformer(input_ids, attn_mask=attn_mask)
            attn_outputs = None
            
        # Determine what to return based on flags
        if return_features:
            if return_attn or self.output_attentions:
                return features, attn_outputs
            else:
                return features
        else:
            if return_attn or self.output_attentions:
                return logits, attn_outputs
            else:
                return logits

    #todo 解决 
    def generate(self, 
                context: torch.Tensor, 
                max_new_tokens: int = 0,
                temperature: float = 1.0,
                top_k: int = 0):
        """
        Autoregressive generation with optional context conditioning
        
        Args:
            context: Context tokens of shape (B, T)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            top_k: Limit sampling to top-k logits (0 = no limit)
            
        Returns:
            Generated token sequence including context
        """
        # Start with provided context
        x = context.clone()
        B, T = x.shape
        
        # Nothing to generate
        if max_new_tokens <= 0:
            return x
            
        # Generate new tokens one by one
        for _ in range(max_new_tokens):
            # Crop context to block_size if needed
            x_cropped = x[:, -self.transformer.embed.max_position_embeddings:]
            
            # Get logits for next token
            if self.output_attentions:
                logits, _, _ = self.transformer(x_cropped)
            else:
                logits, _ = self.transformer(x_cropped)
                
            logits = logits[:, -1, :] # Only need the last position
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
                
            # Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append new token
            x = torch.cat((x, next_token), dim=1)
            
        return x

    def top_k_logits(self, logits, k):
        """Helper to filter logits to only the top k options"""
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out
        
    def get_attention_patterns(self, input_ids, layer_idx=-1):
        """
        Extract attention patterns from a specific layer or all layers
        
        Args:
            input_ids: Input token IDs
            layer_idx: Layer index to extract from (-1 means last layer)
            
        Returns:
            Attention maps
        """
        if not self.output_attentions:
            # Temporarily enable attention output
            old_output_flag = self.transformer.output_attentions
            self.transformer.output_attentions = True
            _, _, attn_outputs = self.transformer(input_ids, return_attention=True)
            self.transformer.output_attentions = old_output_flag
        else:
            _, _, attn_outputs = self.transformer(input_ids, return_attention=True)
            
        if layer_idx == -1:
            return attn_outputs[-1]  # Last layer
        elif layer_idx == 'all':
            return attn_outputs  # All layers
        else:
            # Specific layer
            assert 0 <= layer_idx < len(self.transformer.layers), f"Layer index {layer_idx} out of range"
            return attn_outputs[layer_idx]

