from model import *
from utils import *
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd


BATCH_SIZE = 8
LR = 1e-3
EMBEDDING_DIM = 512
ENCODER_LEN = 1028
DECODER_LEN = 1028
VOCAB_SIZE = 50265
HEADS = 8
LAYERS = 6
EPOCHS = 5
dataset = 'OpenAssistant/oasst1'


class StringDataset(Dataset):
    '''
    Contains all prompts and responses
    '''
    def __init__(self, data: dict) -> None:
        df = pd.DataFrame(data)
        self.prompts = df.loc[df['role'] == 'prompter']['text'].reset_index(drop=True)
        self.responses = df.loc[df['role'] == 'assistant']['text'].reset_index(drop=True)
        
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx], self.responses[idx]


def main():
    # Load data
    train = load_dataset(dataset, split='train')
    test = load_dataset(dataset, split='validation')
    
    train = DataLoader(StringDataset(data=train),
                               batch_size=BATCH_SIZE,
                               shuffle=True,)
    test = StringDataset(data=test)
    print('Data loaded')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(device=device,
                        d_embedding=EMBEDDING_DIM,
                        encoder_max_len=ENCODER_LEN,
                        decoder_max_len=DECODER_LEN,
                        n_heads=HEADS,
                        layers=LAYERS,)
    
    # Train model
    op = optim.AdamW(params=model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    print(f'Training started on {device}')
    
    for _ in range(EPOCHS):
        for prompts, responses in train:
            # Onehot the expected string (softmax probabilities of 1.0)
            tokens = model.tokenizer(responses, max_length=DECODER_LEN, return_tensors='pt', truncation=True, padding=True)
            one_hot = nn.functional.one_hot(tokens['input_ids'].to(device), num_classes=VOCAB_SIZE)
            # Get the true softmax probabilities
            out = model(prompts, inference=False)
            print(out, 'out')
            # Backpropagate
            loss = criterion(out, one_hot)
            loss.backward()
            op.step()
        print('epoch completed')
    
    print(model('hello model'))
    torch.save(model, 'model.pth')

if __name__ == '__main__':
    main()