from model import *
from utils import *
import torch
import torch.nn as nn


def main():
    print('Benchmark #1')
    d_embedding = 10
    pe = PositionalEncoding(max_length=20, 
                            d_embedding=d_embedding)
    dense = FeedForward(d_embedding=d_embedding, 
                        intermed_scale=4)
    res = ResidualConnection(d_embedding=d_embedding)
    att = CrossAttention(d_embedding=d_embedding, n_heads = 2)
    
    x = torch.ones(5, 10)
    x1 = torch.ones(7, 10)
    x = pe(x=x, length=5)
    x = att(x=x, context=x1)
    print(x)


if __name__ == '__main__':
    main()