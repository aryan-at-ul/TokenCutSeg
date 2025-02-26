import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # EMA support
        self.register_buffer('cluster_size', torch.zeros(n_e))
        self.register_buffer('embedding_avg', self.embedding.weight.data.clone())
        self.decay = 0.99
        self.eps = 1e-5
        self.embedding_ema = True

    def forward(self, z):
  
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

      
        if self.training and self.embedding_ema:

            encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=z.device)
            encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)
            
     
            self.cluster_size.data.mul_(self.decay).add_(
                (1 - self.decay) * torch.sum(encodings, 0)
            )
            
    
            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size + self.eps) / (n + self.n_e * self.eps) * n)
            
       
            dw = torch.matmul(encodings.t(), z_flattened)
            self.embedding_avg.data.mul_(self.decay).add_((1 - self.decay) * dw)
            self.embedding.weight.data.copy_(self.embedding_avg / cluster_size.unsqueeze(1))

       
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        z_q = z + (z_q - z).detach()
        
    
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (None, None, min_encoding_indices)