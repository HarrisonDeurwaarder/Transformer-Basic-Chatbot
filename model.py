import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
from utils import *


# ---------------------------
# FEEDFORWARD NETWORK
# ---------------------------

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
        intermed_scale is the scale between the hidden and visible layer
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
        X is the post-attention embeddings of shape [batch, length, embedding_dim]
        '''
        return self.net(x)


# ---------------------------
# ATTENTION MECHANISMS
# ---------------------------


class SelfAttention(nn.Module):
    '''
    Applies scaled dot-product attention to input using learned keys, queries, and values
    '''
    def __init__(self, 
                 d_embedding: int, 
                 n_heads: int,
                 max_len: int = 0,) -> None:
        '''
        Initializes an instance of SelfAttention
        
        Params:
        d_embedding is the embedding dimensions for each token
        n_heads is the number of attention heads, which are concatenated after computation
        max_len is the largest sequence length to be handled
        '''
        super().__init__()
        assert d_embedding % n_heads == 0, 'd_embedding is not divisible by n_heads'
        
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
        
        self.causal_mask = torch.triu(torch.full((max_len, max_len), -torch.inf), diagonal=1).unsqueeze(0)
        self.d_embedding = d_embedding
        self.n_heads = n_heads
        self.d_k = d_embedding // n_heads
    
    def change_device(self, 
                      device: torch.device,) -> None:
        '''
        Ensures that all tensors are properly updated upon device change
        
        Params:
        device is the new torch.device object, generally cuda or cpu
        '''
        self.to(device)
        self.causal_mask = self.causal_mask.to(device)
    
    def forward(self, 
                x: torch.Tensor, 
                use_mask: bool = False,) -> torch.Tensor:
        '''
        Applies pre-calculated positional encodings to X
        
        Params:
        X is the positionally-encoded or dense-layer embeddings of shape [batch, length, embedding_dim]
        in_decoder indicates whether or not to use causal masking to hide future tokens
        '''
        # Apply an optional (decoder-only causal mask to hide future tokens)
        length = x.size(-2)
        mask = self.causal_mask[:length, :length].to(self.causal_mask.device) if use_mask else 0
        
        # Using tuned qkv weights, derive the Q/K/V matrices (shape: [batch, n_heads, length, d_k])
        Q = (x @ self.query.transpose(-2, -1)).view(-1, length, self.n_heads, self.d_k).transpose(-2, -1)
        K = (x @ self.key.transpose(-2, -1)).view(-1, length, self.n_heads, self.d_k).transpose(-2, -1)
        V = (x @ self.value.transpose(-2, -1)).view(-1, length, self.n_heads, self.d_k).transpose(-2, -1)
        qk_dot = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_k) # shape: [batch, n_heads, length, length]
        valued_weights = torch.softmax(qk_dot + mask, dim=-1) @ V # shape: [batch, n_heads, length, d_k]
        concat_heads = valued_weights.transpose(-2, -1).contiguous().view(-1, length, self.d_embedding) # shape: [batch, length, embedding_dim]
        
        # Linear projection applied to concatenated heads
        return concat_heads @ self.out_proj.transpose(-2, -1) # shape: [batch, length, embedding_dim]


class CrossAttention(nn.Module):
    '''
    Applies scaled dot-product attention from decoder inputs to encoder context tensor using learned keys, queries, and values
    '''
    def __init__(self, 
                 d_embedding: int, 
                 n_heads: int) -> None:
        '''
        Initializes an instance of CrossAttention
        
        Params:
        d_embedding is the embedding dimensions for each token
        n_heads is the number of attention heads, which are concatenated after computation
        max_len is the largest sequence length to be handled
        '''
        super().__init__()
        assert d_embedding % n_heads == 0
        
        # Sample weights from identical distribution
        weight = lambda: nn.Parameter(
            torch.empty(d_embedding, d_embedding).uniform_(
                -np.sqrt(3 / d_embedding), 
                np.sqrt(3 / d_embedding))
        )
        self.query = weight()
        self.key = weight()
        self.value = weight()
        self.out_proj = weight()
        
        self.d_embedding = d_embedding
        self.n_heads = n_heads
        self.d_k = d_embedding // n_heads
    
    def forward(self, 
                x: torch.Tensor, 
                context: torch.Tensor,) -> torch.Tensor:
        '''
        Applies pre-calculated positional encodings to X
        
        Params:
        X is the positionally-encoded or dense-layer embeddings of shape [batch, dec_length, embedding_dim]
        context is the context-rich embeddings from the encoder to be attended of shape [batch, enc_length, embedding_dim]
        '''
        length = x.size(-2)
        
        # Using tuned qkv weights, derive the Q/K/V matrices
        Q = (x @ self.query.transpose(-2, -1)).view(-1, length, self.n_heads, self.d_k).transpose(-2, -1) # shape: [batch, n_heads, dec_length, d_k]
        K = (context @ self.key.transpose(-2, -1)).view(-1, length, self.n_heads, self.d_k).transpose(-2, -1) # shape: [batch, n_heads, enc_length, d_k]
        V = (context @ self.value.transpose(-2, -1)).view(-1, length, self.n_heads, self.d_k).transpose(-2, -1) #shape: [batch, n_heads, enc_length, d_k]
        
        qk_dot = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_k) # shape: [batch, n_heads, dec_length, enc_length]
        valued_weights = torch.softmax(qk_dot, dim=-1) @ V # shape: [batch, n_heads, dec_length, d_k]
        concat_heads = valued_weights.transpose(-2, -1).contiguous().view(-1, length, self.d_embedding) # shape: [batch, dec_length, embedding_dim]
        
        # Linear projection applied to concatenated heads
        return concat_heads @ self.out_proj.transpose(-2, -1) # shape: [batch, length, embedding_dim]
    
    
# ---------------------------
# TRANSFORMER ARCHITECTURE
# ---------------------------
    

class Encoder(nn.Module):
    '''
    Runs once for a sequence; generates a context-rich vector of the prompt/equivalent
    '''
    def __init__(self,
                 d_embedding: int,
                 max_len: int = 128,
                 n_heads: int = 8,
                 layers: int = 6,) -> None:
        '''
        Initializes an instance of Encoder
        
        Params:
        d_embedding is the embedding dimensions for each token
        max_len is the largest sequence length to be handled
        n_heads is the number of attention heads, which are concatenated after computation
        layers is the number of repetitions done by the central block
        '''
        super().__init__()
        self.layers = layers
        self.d_embedding = d_embedding
        self.max_len = max_len
        
        # Positional encoding
        self.pe = PositionalEncoding(max_len=max_len, 
                                     d_embedding=d_embedding,)
        # Self-attention layer
        self.att = SelfAttention(d_embedding=d_embedding,
                                 n_heads=n_heads,)
        # FFN
        self.ffn = FeedForward(d_embedding=d_embedding,
                               intermed_scale=4,)
        # Residual connections (add/norm)
        self.res1 = ResidualConnection(d_embedding=d_embedding)
        self.res2 = ResidualConnection(d_embedding=d_embedding)
    
    def change_device(self, 
                      device: torch.device,) -> None:
        '''
        Ensures that all tensors are properly updated upon device change
        
        Params:
        device is the new torch.device object, generally cuda or cpu
        '''
        self.to(device)
        self.pe.change_device(device)
        self.att.change_device(device)
        self.ffn.to(device)
        self.res1.to(device)
        self.res2.to(device)
    
    def forward(self,
                x: torch.Tensor,) -> torch.Tensor:
        '''
        Generates context vector from prompt X
        
        Params:
        X is the embedded tokens of shape [batch, length, embedding_dim]
        '''
        x = self.pe(x)
        # Main encoder block
        for _ in range(self.layers):
            x = self.res1(self.att(x, 
                                   use_mask=False,))
            x = self.res2(self.ffn(x))
        return x
    

class Decoder(nn.Module):
    '''
    Runs until <EOS> token is generated; utilizes context-rich encoded vector and previous tokens to generate new tokens
    '''
    def __init__(self,
                 d_embedding: int,
                 max_len: int = 64,
                 vocab_size: int = 30000,
                 n_heads: int = 8,
                 layers: int = 6,) -> None:
        '''
        Initializes an instance of Decoder
        
        Params:
        d_embedding is the embedding dimensions for each token
        max_len is the largest sequence length to be handled
        vocab_size is the size the list containing possible tokens to generate
        n_heads is the number of attention heads, which are concatenated after computation
        layers is the number of repetitions done by the central block
        '''
        super().__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.d_embedding = d_embedding
        self.max_len = max_len
        
        # Positional encoding
        self.pe = PositionalEncoding(max_len=max_len, 
                                     d_embedding=d_embedding,)
        # Attention layers
        self.att1 = SelfAttention(d_embedding=d_embedding,
                                 n_heads=n_heads,
                                 max_len=max_len,)
        self.att2 = CrossAttention(d_embedding=d_embedding,
                                 n_heads=n_heads,)
        # FFN
        self.ffn = FeedForward(d_embedding=d_embedding,
                               intermed_scale=4,)
        # Residual connections (add/norm)
        self.res1 = ResidualConnection(d_embedding=d_embedding)
        self.res2 = ResidualConnection(d_embedding=d_embedding)
        self.res3 = ResidualConnection(d_embedding=d_embedding)
        # Final transformation
        self.lin = nn.Linear(d_embedding, vocab_size)

    def change_device(self, 
                      device: torch.device,) -> None:
        '''
        Ensures that all tensors are properly updated upon device change
        
        Params:
        device is the new torch.device object, generally cuda or cpu
        '''
        self.to(device)
        self.pe.change_device(device)
        self.att1.change_device(device)
        self.att2.to(device)
        self.ffn.to(device)
        self.res1.to(device)
        self.res2.to(device)
        self.res3.to(device)

    def forward(self,
                x: torch.Tensor,
                context: torch.Tensor,
                temperature: float = 1.0) -> torch.Tensor:
        '''
        Generates a single tokens using X and encoder context
        
        Params:
        X is the previously-generated embedded tokens beginning with <BOS> of shape [batch, length, embedding_dim]
        context is the context-rich embeddings from the encoder to be attended of shape [batch, enc_length, embedding_dim]
        temperature < 0 scales the "unpredictability" of the output, where T > 1 produces less predictable tokens, and T < 1 produces more predictable tokens
        '''
        x = self.pe(x)
        # Main decoder block
        for _ in range(self.layers):
            x = self.res1(self.att1(x, 
                                    use_mask=True,))
            x = self.res2(self.att2(x, 
                                    context=context,))
            x = self.res3(self.ffn(x))
        # Projection to probability distribution
        x = self.lin(x) / temperature
        return torch.softmax(x, dim=1)[-1]
    

class Transformer(nn.Module):
    '''
    A transformer neural network, encodes a prompt and autoregressively generates tokens using attention until <EOS> is reached
    '''
    def __init__(self,
                 device: torch.device,
                 d_embedding: int,
                 encoder_max_len: int = 128,
                 decoder_max_len: int = 64,
                 n_heads: int = 8,
                 layers: int = 6,) -> None:
        '''
        Initializes an instance of Transformer
        
        Params:
        device is the training device, cuda or cpu
        d_embedding is the embedding dimensions for each token
        encoder_max_len is the largest prompt length to be handled
        decoder_max_len is the largest response length to be handled
        vocab_size is the size the list containing possible tokens to generate
        n_heads is the number of attention heads, which are concatenated after computation
        layers is the number of repetitions done by the central block
        max_len is the largest sequence length to be handled
        '''
        super().__init__()
        
        # Embedding/tokenizing models
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
        self.vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=d_embedding).to(device)
        
        self.d_embedding = d_embedding
        # P.E. hyperparams
        self.enc_len = encoder_max_len
        self.dec_len = decoder_max_len
        # Main model hyperparams
        self.heads = n_heads
        self.layers = layers
        # Enc/dec objects
        self.enc = Encoder(d_embedding=d_embedding,
                           max_len=encoder_max_len,
                           n_heads=n_heads,
                           layers=layers,)
        self.dec = Decoder(d_embedding=d_embedding,
                           max_len=decoder_max_len,
                           vocab_size=self.vocab_size,
                           n_heads=n_heads,
                           layers=layers,)
        self.change_device(device)
        
    def change_device(self, 
                      device: torch.device,) -> None:
        '''
        Ensures that all tensors are properly updated upon device change
        
        Params:
        device is the new torch.device object, generally cuda or cpu
        '''
        self.device = device
        self.to(device)
        self.embedding = self.embedding.to(device)
        self.enc.change_device(device)
        self.dec.change_device(device)
    
    def embed(self,
              tokens: torch.Tensor,) -> torch.Tensor:
        '''
        Turns a tokenized string (ids) into an embedding matrix of shape [batch, length, embedding_dim]
        
        Params:
        tokens is the id-tensor input to the transformer of shape [batch, length, vocab_size]
        '''
        # Get embedding vectors
        return self.embedding(tokens).squeeze(0)
    
    @timer
    def forward(self,
                x: str,
                temperature: float = 1.0,
                inference: bool = True) -> torch.Tensor:
        '''
        Passes string-prompt X through the transformer
        
        Params:
        X is a string containing the prompt
        temperature < 0 scales the "unpredictability" of the output, where T > 1 produces less predictable tokens, and T < 1 produces more predictable tokens
        inference determines whether strings or gradient-ready softmax probabilities are returned
        '''
        # Tokenize
        tokens = self.tokenizer(x, max_length=self.enc.max_len, return_tensors='pt', truncation=True, padding=True)
        # Get context vector from encoder
        x = self.embed(tokens=tokens['input_ids'].to(self.device))
        print(x.shape)
        context = self.enc(x)
        print(context.shape)
        
        # ids are converted to a string, embedded_ids are fed back into the model
        ids = torch.Tensor(torch.zeros(x.size(0)) if x.size(0) > 1 else [0]).int().to(self.device) # Handle batches vs. single-prompt inference
        embedded_ids = self.embed(ids)
        probs = torch.Tensor()
        
        # Autoregressively run the decoder until an EOS is produced (id=2)
        while self.tokenizer.decode(ids[-1]) != 2 and ids.size(0) < self.dec.max_len:
            # Reevaluate embeddings
            embedded_ids = self.embed(ids)
            # Softmax probabilities
            probs = torch.cat([probs, self.dec(embedded_ids, context=context, temperature=temperature)])
            # Sampled IDs
            ids = torch.cat([ids, torch.multinomial(probs[-1], num_samples=1, replacement=False)])
            
        # Convert ids to string for easy inference, else return raw ids
        if inference:
            return self.tokenizer.decode(ids[1:])
        
        return probs