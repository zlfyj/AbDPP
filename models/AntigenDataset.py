import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from configuration_antibody import AA_VOCAB
import numpy as np
from torch.utils.data import Dataset, DataLoader
from configuration_antibody import configuration
import pdb
from tqdm import tqdm 
from modeling_antibody import get_attn_pad_mask
from math import ceil

class antigen_dataset(Dataset):
    def __init__(self,
                 config:configuration,
                 data_path='../data/protein.csv',
                 train = True,
                 rate=0.8
                 ) -> None:
        super().__init__()
        self.config = config
        df = pd.read_csv(data_path)
        if train:
            self.data = df.iloc[:int(df.shape[0]*rate)]
        else:
            self.data = df.iloc[int(df.shape[0]*rate):]

    def padding(self,seq,maxlen):
        if len(seq) > maxlen:
            return seq[:maxlen]
        else:
            return torch.cat([seq,torch.zeros(maxlen-len(seq))]).long()
    
    def add_loards(self,seq):
        
        lord = torch.tensor([AA_VOCAB['X']])
        lords = []
        for i in range(self.config.segments):
            lord_split = i*self.config.segment_size+1+i
            lords.append(lord_split)
            seq = torch.cat([seq[:lord_split],lord,seq[lord_split:]])
        return seq
        
    def __getitem__(self, index):
        seq = torch.tensor([AA_VOCAB[aa] for aa in self.data['Sequence'].iloc[index]])
        label = torch.tensor(self.data['SUPFAM'].iloc[index])
        Blord = torch.tensor([AA_VOCAB['-']])
        seq = torch.cat([Blord,seq])
        
        seq = self.padding(seq,self.config.max_position_embeddings-self.config.segments)  ## 5 lords
        if self.config.segment:
            seq = self.add_loards(seq)
            length = len(self.data['Sequence'].iloc[index])
            length = ceil(length/self.config.segment_size) + length + 1
            mask = torch.ones(length).view(-1)
            mask = self.padding(mask,self.config.max_position_embeddings).view(1,-1)
            mask = get_attn_pad_mask(mask,mask)
        else:
            mask = get_attn_pad_mask(seq.unsqueeze(0),seq.unsqueeze(0))
        
        return (seq,mask.squeeze(0),label)

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    config = configuration()
    setattr(config,'segment',True)
    setattr(config,'max_position_embeddings',1024)
    setattr(config,'segments',4)
    setattr(config,'segment_size',int(config.max_position_embeddings/config.segments))
    setattr(config,'max_position_embeddings',1024+config.segments+1)
    dataset = antigen_dataset(config)
    dataset[6]
        