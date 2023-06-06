from modeling_antibody import *
from AntiTranslate_dataset import antiTranslate_dataset
import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb
from torch.utils.data import DataLoader
from configuration_antibody import configuration


class AntiEncoder_T(nn.Module):
    def __init__(self,config,use_cuda=False) -> None:
        super().__init__()
        self.config = config
        self.use_cuda = use_cuda
        self.embedding = AntiEmbeddings(config)
        self.encoder = AntiEncoder(config)
    def forward(self,antibodies,type_token,mask):
        
        x = self.embedding(antibodies,type_token)
        x = self.encoder(x,attention_mask = mask)
        return x
        
class AntiDecoder_T(nn.Module):
    def __init__(self,config,use_cuda=False) -> None:
        super().__init__()
        self.config = config
        self.use_cuda = use_cuda
        self.embedding = AntiEmbeddings(config)
        setattr(self.config,'is_decoder',True)
        setattr(self.config,'add_cross_attention',True)

        self.decoder = AntiEncoder(config)
    def forward(self,cdr,cdr_type,cdr_mask,encoder_hidden_states,encoder_attention_mask):
        x = self.embedding(cdr,cdr_type)
        
        x = self.decoder(x,attention_mask=cdr_mask,encoder_hidden_states=encoder_hidden_states,encoder_attention_mask=encoder_attention_mask)
        return x[0]

class antigen_convert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convert = nn.Sequential(
                    nn.Linear(1280,768*3,bias=True),
                    nn.ReLU(),
                    nn.Linear(768*3,768),
                    nn.ReLU()
                )
    def forward(self,x):
      
        return self.convert(x)
        

class AntiTranslate(nn.Module):
    def __init__(self,encoder_config,decoder_config,device='cpu') -> None:
        super().__init__()

        self.use_cuda = True
        if device == 'cpu':
            self.use_cuda = False
        self.encoder = AntiEncoder_T(encoder_config,self.use_cuda)
        self.decoder = AntiDecoder_T(decoder_config,self.use_cuda)
        self.dense = nn.Sequential(nn.Linear(decoder_config.hidden_size,256,256),
                                    nn.ReLU(),
                                    nn.Linear(256,128),
                                    nn.ReLU(),
                                    nn.Linear(128,22)
                                    )
       
    def forward(self,x,antigen=None):
        antibodies, token_type, fcf, cdr,cdr_type = x[0], x[1], x[2], x[3], x[4]
        
        mask = get_attn_pad_mask(antibodies,antibodies)
        cdr_mask = get_attn_subsequence_mask(cdr.cpu())
        
        encoder_attention_mask = get_attn_pad_mask(cdr,antibodies)
        
        if self.use_cuda:
            antibodies, token_type, cdr,cdr_type,mask,cdr_mask,encoder_attention_mask = antibodies.cuda(), token_type.cuda(), cdr.cuda(),cdr_type.cuda(),mask.cuda(),cdr_mask.cuda(),encoder_attention_mask.cuda()
        
        encoder_output= self.encoder(antibodies,token_type,mask)[0] 
        
        if antigen != None:
            encoder_output = torch.cat([encoder_output,antigen.unsqueeze(1)],dim=1)
            encoder_attention_mask = torch.cat([encoder_attention_mask,torch.zeros(encoder_attention_mask.shape)[:,:,:,1].unsqueeze(-1).cuda()],dim=-1)
        
        decoder_output = self.decoder(cdr,cdr_type,cdr_mask,encoder_output,encoder_attention_mask=encoder_attention_mask)
        output= self.dense(decoder_output)[:,:-1]
        
        return output
        

if __name__ == '__main__':
    config_e = configuration()
    config_d = configuration()
    dataset = antiTranslate_dataset(config_d,datapath='../data/met/met_pos.csv')
    antigen_config = configuration()
    

    setattr(config_e,'max_position_embeddings',141)
    setattr(config_d,'max_position_embeddings',141)

    antitranslate = AntiTranslate(config_e,config_d,True)
    
    #antitranslate = torch.nn.DataParallel(antitranslate).cuda()
    weight = torch.load('../ckpts/pretrain/AntiTranslate50,pre_train,cdr3,6,6,512+6e-05.pth')
    antitranslate.load_state_dict({k.replace('module.', ''): v for k, v in weight.items()})

    dataloader = DataLoader(dataset,batch_size=32)
    for x,antigen in dataloader:
        antitranslate(x,antigen)
        pdb.set_trace()