import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from configuration_antibody import AA_VOCAB
import numpy as np
from torch.utils.data import Dataset, DataLoader
from configuration_antibody import configuration
import pdb
import esm
import os
class antiTranslate_dataset(Dataset):
    def __init__(self,
                config,
                datapath,
                train = True,
                region='cdr3',
                split=0.8,
                antigen_esm=False,
                data=None) -> None:
        super().__init__()
        """"
            config: model configuration
            datapaht
            train: True( train model) False (val model)

            data.coloumns :['seq', 'H-FR1', 'H-CDR1', 'H-FR2',
                    'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4']
            region : CDR [1-3]
        """
        self.region = region
        self.config = config
        self.train = train
        if data != None:
            self.data = data
        else:
            self.data = pd.read_csv(datapath)
        self.train_num = int(split*self.data.shape[0])
        self.antigen_esm = antigen_esm
        self.antigen_model,alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = alphabet.get_batch_converter()
    def padding(self,seq,maxlen):
        if len(seq) > maxlen:
            return seq[:maxlen]
        else:
            return torch.cat([seq,torch.zeros(maxlen-len(seq))]).long()

    def input_seq(self,vh):
        vh = self.padding(vh,self.config.max_position_embeddings-1)
        cls = torch.tensor([self.config.AA_VOCAB['*']],dtype=torch.long)
        return torch.cat([cls,vh])

    def __getitem__(self, index):
        if not self.train:
            index += self.train_num
        data = self.data.iloc[index]

        HF1 = [1 for _ in range(len(data['H-FR1']))]
        HCDR1 = [2 for _ in range(len(data['H-CDR1']))]
        HF2 = [1 for _ in range(len(data['H-FR2']))]
        HCDR2 = [3 for _ in range(len(data['H-CDR2']))]
        HF3 = [1 for _ in range(len(data['H-FR3']))]
        HCDR3 = [4 for _ in range(len(data['H-CDR3']))]
        HF4 = [1 for _ in range(len(data['H-FR4']))]

        if self.region == 'cdr1':
            seq = data['H-FR1'] + "%" + data['H-FR2']
            cdr = "%" + data['H-CDR1'] + "X" 
            token_type = torch.tensor(list([0] + HF1 + [5] + HF2))
            fcf = len(HF1)
        elif self.region == 'cdr2':
            seq = data['H-FR1'] + data['H-CDR1'] + data['H-FR2'] + "$"+ data['H-FR3']
            cdr = "$"+ data['H-CDR2'] + "X" 
            token_type =  torch.tensor(list([0] + HF1 +HCDR1+ HF2 + [6] + HF3))
            fcf = len(data['H-FR1']) + len(data['H-CDR1']) + len(data['H-FR2'])
        elif self.region == 'cdr3':
            seq = data['H-FR1'] + data['H-CDR1'] + data['H-FR2'] + data['H-CDR2'] + data['H-FR3'] + "@" + data['H-FR4']
            cdr = "@" + data['H-CDR3'] + "X" 
            token_type = torch.tensor((list([0] + HF1 +HCDR1+ HF2 + HCDR2 + HF3 + [7] + HF4)))
            fcf = len(data['H-FR1']) + len(data['H-CDR1']) + len(data['H-FR2']) + len(data['H-CDR2']) + len(data['H-FR3'])
        fcf += 1 
        
        aaseq = torch.tensor([AA_VOCAB[aa] for aa in seq])
        aacdr = torch.tensor([AA_VOCAB[aa] for aa in cdr])

        token_type = self.padding(token_type,self.config.max_position_embeddings)
        aaseq = self.input_seq(aaseq)

        cdr_type = torch.tensor([0]+[4]*(aacdr.shape[0]-1))
        aacdr = self.padding(aacdr,self.config.maxlen_HCDR3)
        cdr_type = self.padding(cdr_type,self.config.maxlen_HCDR3)
        ###############################################
        if self.antigen_esm == True:
        #######################ESM##################
            if not os.path.exists('../antigen_esm/'+self.data.iloc[index]['antigen']+'.pt'):
                
                #antigen = self.padding_esm(self.data['Antigen Sequence'].iloc[index],self.antigen_config.max_position_embeddings)
                antigen = self.data['Antigen Sequence'].iloc[index]
                antigen = [('antigen',antigen)]
                batch_labels, batch_strs, antigen = self.batch_converter(antigen)
                with torch.no_grad():
                    self.antigen_model = self.antigen_model.eval()
                    #antigen = antigen.cuda()
                    antigen = self.antigen_model(antigen.squeeze(1), repr_layers=[33], return_contacts=True)
                    antigen = antigen['representations'][33].norm(dim=1).view(-1)
                 
                    torch.save(antigen,'../antigen_esm/'+self.data.iloc[index]['antigen']+'.pt')
        ############################################
            antigen = torch.load('../antigen_esm/'+self.data.iloc[index]['antigen']+'.pt')
            return (aaseq,token_type,fcf, aacdr, cdr_type,antigen)
        return (aaseq,token_type,fcf, aacdr, cdr_type)


    def __len__(self):
        if self.train:
            return self.train_num
        else:
            return len(self.data)- self.train_num

def remake_data(antibodies,
                token_type,
                fcf,
                cdr,
                region):
    """
    cdr[1-3] : [2-4]
    speacial token cdr[1-3] = [5-7]
    """
    antibodies = antibodies.clone()
    token_type = token_type.clone()
    if region == 'cdr1':
        residue_type = [2]
    elif region == 'cdr2':
        residue_type = [3]
    else:
        residue_type = [4]
    def find_end(seq)->int:
            # find end token
            for i in range(len(seq)):
                if seq[i] == 0:
                    return i
            return len(seq)
    for i in range(antibodies.shape[0]):
        end_index = find_end(cdr[i])
        cdr_token_type = torch.tensor(residue_type * end_index)
        antibodies[i] = torch.concat([antibodies[i][:fcf[i]], cdr[i][:end_index], antibodies[i][fcf[i]+1:-end_index+1]])
        token_type[i] = torch.concat([token_type[i][:fcf[i]], cdr_token_type, token_type[i][fcf[i]+1:-end_index+1]])
    return [antibodies,token_type]



if __name__ == '__main__':
    config = configuration()
    setattr(config,'max_position_embeddings',141)
    region = 'cdr3'
    train_dataset = antiTranslate_dataset(config,datapath='../data/neturalize/pos.csv',region=region)
    train_dataset[0]
    dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True)
    for x in dataloader:
        
        antibodies, token_type = remake_data(x[0],x[1],x[2],x[3],region=region)
        pdb.set_trace()
        

    
