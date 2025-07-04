import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import *


class PositionalEncoding(nn.Module):
    '''
    Encodes position into the embeddings, which are not naturally in postion-agnostic transformers
    '''
    def __init__(self, 
                 max_length: int, 
                 d_embedding: int,) -> None:
        '''
        Initializes an instance of PositionalEncoding
        
        Params:
        max-length is the largest sequence length to be handled
        d_embedding is the embedding dimensions for each token
        '''
        super().__init__()
        self.pe = torch.zeros(max_length, d_embedding)
        
        pos = torch.arange(0, max_length).unsqueeze(1)
        emb = torch.arange(0, d_embedding, 2).unsqueeze(0)
        inner_term = pos / torch.pow(10000, emb / d_embedding)
        
        # Apply alternating sin/cos to quotient
        self.pe[:, 0::2] = torch.sin(inner_term)
        self.pe[:, 1::2] = torch.cos(inner_term)
    
    def forward(self, 
                x: torch.Tensor, 
                length: int,) -> torch.Tensor:
        '''
        Applies pre-calculated positional encodings to X
        
        Params:
        X is the unaltered input embeddings [length, embedding-dim]
        '''
        return x + self.pe[:length]


class MultiHeadAttention(nn.Module):
    '''
    Applies scaled dot-product attention to input using learned keys, queries, and values
    '''
    def __init__(self, 
                 d_embedding: int, 
                 n_heads: int,) -> None:
        '''
        Initializes an instance of MultiHeadAttention
        
        Params:
        d_embedding is the embedding dimensions for each token
        n_heads is the number of attention heads, which are concatenated after computation
        '''
        super().__init__()
        assert d_embedding % n_heads == 0
        
        # Clone weights from identical distribution
        weight = lambda: nn.Parameter(
            torch.empty(d_embedding, d_embedding).uniform_(
                -np.sqrt(3 / d_embedding), 
                np.sqrt(3 / d_embedding))
        )
        self.query = weight()
        self.key = weight()
        self.value = weight()
        self.out_proj = weight()
        
        self.causal_mask = lambda length: torch.triu(torch.full((length, length), -torch.inf), diagonal=1).unsqueeze(0)
        self.d_embedding = d_embedding
        self.n_heads = n_heads
        self.d_k = d_embedding // n_heads
    
    def forward(self, x: torch.Tensor, in_decoder: bool = False) -> torch.Tensor:
        '''
        Applies pre-calculated positional encodings to X
        
        Params:
        X is the positionally-encoded or dense-layer embeddings of shape [length, embedding_dim]
        in_decoder indicates whether or not to use causal masking to hide future tokens
        '''
        # Apply an optional (decoder-only causal mask to hide future tokens)
        mask = self.causal_mask(x.size(0)) if in_decoder else 0
        
        # Using tuned qkv weights, derive the Q/K/V matrices (shape: [n_heads, length, d_k])
        Q = (x @ self.query.transpose(-2, -1)).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = (x @ self.key.transpose(-2, -1)).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = (x @ self.value.transpose(-2, -1)).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        
        qk_dot = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_k) # shape: [n_heads, length, length]
        valued_weights = torch.softmax(qk_dot + mask, dim=-1) @ V.transpose(-2, -1) # shape: [n_heads, length, d_k]
        concat_heads = valued_weights.transpose(0, 1).contiguous().view(-1, self.d_embedding) # shape: [length, embedding_dim]
        
        # Linear projection applied to concatenated heads
        return concat_heads @ self.out_proj.transpose(-2, -1) # shape: [length, embedding_dim]


class ResidualConnection(nn.Module):
    '''
    Adds a normalized embedding vector to every token
    '''
    def __init__(self,
                 d_embedding: int,) -> None:
        '''
        Initializes an instance of ResidualConnection
        
        Params:
        d_embedding is the embedding dimensions for each token
        '''
        super().__init__()
        # Scale and shift
        self.gamma = nn.Parameter(torch.ones(d_embedding))
        self.beta = nn.Parameter(torch.zeros(d_embedding))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Adds the normalized vector
        
        Params:
        X is the post-attention or post-ffn embeddings of shape [length, embedding-dim]
        '''
        epsilon = 1e-5
        mean = torch.mean(x, dim=-1).unsqueeze(1)
        # Variance = std^2
        var = torch.mean(torch.square(x - mean))
        # Epsilon term prevents div-by-zero
        std = torch.sqrt(var + epsilon)
        
        norm = (x - mean) / std
        # Learnable weight/bias are applied
        return self.gamma * norm + self.beta


class FeedForward(nn.Module):
    '''
    2-layer feedforward neural network sublayer, applied to each token
    '''
    def __init__(self, 
                 d_embedding: int, 
                 intermed_scale: int = 4,) -> None:
        '''
        Initializes an instance of FeedForward
        
        Params:
        d_embedding is the embedding dimensions for each token
        '''
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embedding, d_embedding * intermed_scale),
            nn.ReLU(),
            nn.Linear(d_embedding * intermed_scale, d_embedding)
        )
    
    def forward(self, 
                x: torch.Tensor,) -> torch.Tensor:
        '''
        Pass X through dense sublayer
        X is the post-attention embeddings of shape [length, embedding_dim]
        '''
        return self.net(x)


class CrossAttention(nn.Module):
    '''
    Applies scaled dot-product attention from decoder inputs to encoder context tensor using learned keys, queries, and values
    '''
    def __init__(self, 
                 d_embedding: int, 
                 n_heads: int,) -> None:
        '''
        Initializes an instance of CrossAttention
        
        Params:
        d_embedding is the embedding dimensions for each token
        n_heads is the number of attention heads, which are concatenated after computation
        '''
        super().__init__()
        assert d_embedding % n_heads == 0
        
        # Clone weights from identical distribution
        weight = lambda: nn.Parameter(
            torch.empty(d_embedding, d_embedding).uniform_(
                -np.sqrt(3 / d_embedding), 
                np.sqrt(3 / d_embedding))
        )
        self.query = weight()
        self.key = weight()
        self.value = weight()
        self.out_proj = weight()
        
        self.causal_mask = lambda length: torch.triu(torch.full((length, length), -torch.inf), diagonal=1).unsqueeze(0)
        self.d_embedding = d_embedding
        self.n_heads = n_heads
        self.d_k = d_embedding // n_heads
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        '''
        Applies pre-calculated positional encodings to X
        
        Params:
        X is the positionally-encoded or dense-layer embeddings of shape [dec_length, embedding_dim]
        context is the context-rich embeddings from the encoder to be attended of shape [enc_length, embedding_dim]
        '''
        # Using tuned qkv weights, derive the Q/K/V matrices
        Q = (x @ self.query.transpose(-2, -1)).view(-1, self.n_heads, self.d_k).transpose(0, 1) # shape: [n_heads, dec_length, d_k]
        K = (context @ self.key.transpose(-2, -1)).view(-1, self.n_heads, self.d_k).transpose(0, 1) # shape: [n_heads, enc_length, d_k]
        V = (context @ self.value.transpose(-2, -1)).view(-1, self.n_heads, self.d_k).transpose(0, 1) #shape: [n_heads, enc_length, d_k]
        
        qk_dot = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_k) # shape: [n_heads, dec_length, enc_length]
        print(torch.softmax(qk_dot, dim=-1).size(), V.transpose(-2, -1).size())
        valued_weights = torch.softmax(qk_dot, dim=-1) @ V # shape: [n_heads, dec_length, d_k]
        concat_heads = valued_weights.transpose(0, 1).contiguous().view(-1, self.d_embedding) # shape: [dec_length, embedding_dim]
        
        # Linear projection applied to concatenated heads
        return concat_heads @ self.out_proj.transpose(-2, -1) # shape: [length, embedding_dim]
    

class Encoder(nn.Module):
    
    def __init__(self,) -> None:
        '''
        Initializes an instance of Encoder
        
        Params:
        d_embedding is the embedding dimensions for each token
        '''
        super().__init__()
    
    def forward(self,) -> torch.Tensor:
        '''
        Applies pre-calculated positional encodings to X
        
        Params:
        
        '''
        pass
    

class Decoder(nn.Module):
    
    def __init__(self,
                 d_embedding,) -> None:
        '''
        Initializes an instance of Decoder
        
        Params:
        d_embedding is the embedding dimensions for each token
        '''
        super().__init__()
    
    def forward(self,) -> torch.Tensor:
        '''
        Applies pre-calculated positional encodings to X
        
        Params:
        
        '''
        pass
    

class Transformer(nn.Module):
    
    def __init__(self,) -> None:
        '''
        Initializes an instance of Transformer
        
        Params:
        d_embedding is the embedding dimensions for each token
        '''
        super().__init__()
    
    def forward(self,) -> torch.Tensor:
        '''
        Applies pre-calculated positional encodings to X
        
        Params:
        
        '''
        pass