from model import *
from utils import *
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


BATCH_SIZE = 8
LR = 1e-3
EMBEDDING_DIM = 512
ENCODER_LEN = 128
DECODER_LEN = 64
VOCAB_SIZE = 30000
HEADS = 8
LAYERS = 6
EPOCHS = 5


class StringDataset(Dataset):
    '''
    Contains all prompts and responses
    '''
    def __init__(self, data: dict) -> None:
        self.data = data
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> str:
        return self.data[idx]


def main():
    # Load data
    train = load_dataset('blended_skill_talk', split='train')
    test = load_dataset('blended_skill_talk', split='test')
    
    train = DataLoader(StringDataset(data={'prompt': train['context'][-1], 
                                        'response': train['response']}),
                               batch_size=BATCH_SIZE,
                               shuffle=True,)
    test = StringDataset(data={'prompt': test['context'][-1], 
                                       'response': test['response']})
    
    model = Transformer(encoder=Encoder(d_embedding=EMBEDDING_DIM,
                                        max_len=ENCODER_LEN,
                                        n_heads=HEADS,
                                        layers=LAYERS,),
                        decoder=Decoder(d_embedding=EMBEDDING_DIM,
                                        max_len=DECODER_LEN,
                                        vocab_size=VOCAB_SIZE,
                                        n_heads=HEADS,
                                        layers=LAYERS,))
    # Train model
    op = optim.AdamW(params=model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(EPOCHS):
        for batch in train:
            # Onehot the expected string (softmax probabilities of 1.0)
            tokens = model.tokenizer(batch['response'], max_length=DECODER_LEN, return_tensors='pt', truncation=True, padding=True)
            one_hot = nn.functional.one_hot(tokens, num_classes=VOCAB_SIZE)
            # Get the true softmax probabilities
            out = model(batch['prompt'], inference=False)
            # Backpropagate
            loss = criterion(out, one_hot)
            loss.backward()
            op.step()
    
    print(model('hello model'))
    torch.save(model, 'model.pth')

if __name__ == '__main__':
    main()