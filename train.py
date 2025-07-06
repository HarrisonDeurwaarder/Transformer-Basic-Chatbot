from model import *
from utils import *
import torch
import torch.nn as nn
from transformers import AutoTokenizer


def main():
    print('Benchmark #1')
    dim = 24
    t = Transformer(Encoder(dim),
                    Decoder(dim))
    print(t('i have apples'))


if __name__ == '__main__':
    main()