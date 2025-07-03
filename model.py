import torch
import torch.nn as nn
import torch.optim as optim


class PositionalEncoding(nn.Module):
    # Pre-calculate sinusoidal encodings
    def __init__(self, max_length: int, d_embedding: int) -> None:
        super().__init__()
        self.pe = torch.zeros(max_length, d_embedding)
        self.d_embed = d_embedding
        
        pos = torch.arange(0, max_length).unsqueeze(1)
        emb = torch.arange(0, d_embedding, 2).unsqueeze(0)
        inner_term = pos / torch.pow(10000, emb / d_embedding)
        
        # Apply alternating sin/cos to quotient
        self.pe[:, 0::2] = torch.sin(inner_term)
        self.pe[:, 1::2] = torch.cos(inner_term)
    
    # Apply positional encoding matrix to sequence of X length
    # Where x is of dimensions [Length, embedding dim]
    def forward(self, x: torch.Tensor, length: int):
        '''
        Applies pre-calculated positional encodings to X
        X is of shape [length, embedding-dim]
        '''
        print(x.size(), self.pe[0, :length].size())
        return x + self.pe[:length]


class MultiHeadedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        pass


class AddNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        pass


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        pass
    

class CausalMask(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        pass
    

class CrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        pass
    

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        pass
    

class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        pass
    

class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        pass