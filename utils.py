import time
import torch
import torch.nn as nn


def timer(func):
    '''
    Times a function call and displays the result
    '''
    def wrapper(*args, **kwargs):
        # Get time before function call
        start = time.time()
        result = func(*args, **kwargs)
        # Print result
        print(f'\033[90m{func.__name__}() took {time.time() - start:.5f} seconds.\033[0m')
        return result
    return wrapper


def info(tokenizer) -> None:
    '''
    Displays token information for a specific tokenizer
    
    Params:
    tokenizer is the tokenizing model from transformers.AutoTokenizer
    '''
    print(f'---\n{tokenizer.name_or_path}\nCLS: {tokenizer.cls_token}\nSEP: {tokenizer.sep_token}\nBOS: {tokenizer.bos_token}\nEOS: {tokenizer.eos_token}\n---')


class PositionalEncoding(nn.Module):
    '''
    Encodes position, which are not naturally present, into embeddings (position-agnostic)
    '''
    def __init__(self,
                 d_embedding: int,
                 max_len: int,) -> None:
        '''
        Initializes an instance of PositionalEncoding
        
        Params:
        max_len is the largest sequence length to be handled
        d_embedding is the embedding dimensions for each token
        '''
        super().__init__()
        self.max_len = max_len
        self.pe = torch.zeros(max_len, d_embedding)
        
        pos = torch.arange(0, max_len).unsqueeze(1)
        emb = torch.arange(0, d_embedding, 2).unsqueeze(0)
        inner_term = pos / torch.pow(10000, emb / d_embedding)
        
        # Apply alternating sin/cos to quotient
        self.pe[:, 0::2] = torch.sin(inner_term)
        self.pe[:, 1::2] = torch.cos(inner_term)
    
    def forward(self, 
                x: torch.Tensor,) -> torch.Tensor:
        '''
        Applies pre-calculated positional encodings to X
        
        Params:
        X is the unaltered input embeddings [length, embedding-dim]
        '''
        assert x.size(0) <= self.max_len, 'Token length exceeds maximum'
        return x + self.pe[:x.size(0)]


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
    
    def forward(self, 
                x: torch.Tensor,) -> torch.Tensor:
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