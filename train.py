from model import *
import torch
import torch.nn as nn


def main():
    print('Benchmark #1')
    pe = PositionalEncoding(max_length=20, d_embedding=10)
    x = torch.ones(5, 10)
    print(pe(x=x, length=5))


if __name__ == '__main__':
    main()