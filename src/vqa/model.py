import torch
import torch.nn as nn
from .vqagan import VQGAN

class VQA(nn.Module):
    def __init__(self, n_embed, embed_dim, hidden_channels=128, n_res_blocks=2):
        super().__init__()
        self.vqgan = VQGAN(n_embed, embed_dim, hidden_channels, n_res_blocks)
        
    def encode(self, x):
        return self.vqgan.encode(x)
        
    def decode(self, z):
        return self.vqgan.decode(z)
        
    def forward(self, x):
        return self.vqgan(x)

    def get_codebook_indices(self, x):
        quant, _, info = self.encode(x)
        return info[2]
        
    def get_codebook_entry(self, indices, shape=None):
        return self.vqgan.quantize.get_codebook_entry(indices, shape)