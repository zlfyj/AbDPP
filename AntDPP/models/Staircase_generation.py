import torch 
import torch.nn as nn
import numpy as np
from AntiTranslate import AntiTranslate, antigen_convert
from modeling_antibody import *
from AntiTranslate_dataset import antiTranslate_dataset
from configuration_antibody import AA_VOCAB 
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from Antibody_Antigen_model import antigen_antibody
from torch.distributions.categorical import Categorical
import pandas as pd
import torch.nn.functional as F


class starecase_generation():
    def __init__(self,
                cdr1_state_path=None,
                cdr2_state_path=None,
                cdr3_state_path=None,
                antigen_state_path=None,
                gpu = '0',
                gen_pos_path = '../data/neturalize/gen_pos.csv',
                esm=False,
                sample=False,
                antigen=False) -> None:
        config_e = configuration()
        config_d = configuration()
        setattr(config_e,'max_position_embeddings',141)
        setattr(config_e,'num_hidden_layers',6)
        setattr(config_e,'region_info',True)
        setattr(config_e,'type_embedding',True)

        setattr(config_d,'max_position_embeddings',141)
        setattr(config_d,'num_hidden_layers',6) 
        setattr(config_d,'region_info',True)
        setattr(config_d,'type_embedding',True)
        if esm:
            setattr(config_d,'hidden_size',1280)
            setattr(config_d,'num_attention_heads',16)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        self.esm = esm
        self.sample = sample
        self.antigen = antigen
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'gpu'

        if cdr1_state_path==None and cdr2_state_path==None and cdr3_state_path==None:
            raise('no ckpts')
        
        self.cdr1_translate = AntiTranslate(config_e,config_d,device)
        self.cdr2_translate = AntiTranslate(config_e,config_d,device)
        self.cdr3_translate = AntiTranslate(config_e,config_d,device)
        self.antigen_model = antigen_convert()
           
        if device == 'gpu':
            if cdr1_state_path!= None:
                self.cdr1_translate = torch.nn.DataParallel(self.cdr1_translate).cuda().eval()
            if cdr2_state_path!= None:  
                self.cdr2_translate = torch.nn.DataParallel(self.cdr2_translate).cuda().eval()
            if cdr3_state_path!= None:
                self.cdr3_translate = torch.nn.DataParallel(self.cdr3_translate).cuda().eval()
            if antigen_state_path!=None:
                self.antigen_state_path = antigen_state_path
                
        
        if cdr1_state_path!= None:
            cdr1_state = torch.load(cdr1_state_path)
            self.cdr1_translate.load_state_dict(cdr1_state)
        if cdr2_state_path!= None:
            cdr2_state = torch.load(cdr2_state_path)
            self.cdr2_translate.load_state_dict(cdr2_state)
        if cdr3_state_path!= None:
            
            cdr3_state = torch.load(cdr3_state_path)
            self.cdr3_translate.load_state_dict(cdr3_state)
    
        self.gen_pos_path = gen_pos_path

        self.reverse_AA_VOVAB = {v:k for k,v in AA_VOCAB.items()}

    def setup(self,region):
        config = configuration()
        setattr(config,'max_position_embeddings',141)
        setattr(config,'num_hidden_layers',12)
        if self.antigen:
            dataset = antiTranslate_dataset(config,self.gen_pos_path,region=region,split=1,antigen_esm=True)
        else:    
            dataset = antiTranslate_dataset(config,self.gen_pos_path,region=region,split=1)
        def GCD(number,floor):
            for i in range(floor,0,-1):
                if number%i == 0:
                    return i
        batch_size = GCD(dataset.__len__(),1024)
        batch_size = 128
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
        generator = None
        cdr_pre = None
        cdr_region = None
        

        if self.antigen:
            self.antigen_model = torch.nn.DataParallel(self.antigen_model).cuda().eval()

        
        def find_end(seq):
            for i in range(len(seq)):
                if seq[i] == 0 or seq[i] == 21:
                    return i
            return i
        cdrs = []
        print('generating:',region)
        with torch.no_grad():
            for data in tqdm(dataloader):
                """
                "@": 25,    # <HCDR3>
                "$": 26,    # <HCDR2>
                "%": 27,    # <HCDR1>
                """
                batch_size = data[0].shape[0]
                if region == 'cdr1':
                    generator = self.cdr1_translate
                    if self.antigen:
                        self.antigen_model = antigen_convert()
                        self.antigen_model.load_state_dict(torch.load(antigen_state_path[0]))
                    cdr_pre = torch.tensor([AA_VOCAB['%']]*batch_size)
                    cdr_region = torch.tensor([2]*batch_size)
                elif region == 'cdr2':
                    generator = self.cdr2_translate
                    if self.antigen:
                        self.antigen_model = antigen_convert()
                        self.antigen_model.load_state_dict(torch.load(antigen_state_path[1]))
                
                    cdr_pre = torch.tensor([AA_VOCAB['$']]*batch_size)
                    cdr_region = torch.tensor([3]*batch_size)
                elif region == 'cdr3':
                    generator = self.cdr3_translate
                    if self.antigen:
                        self.antigen_model = antigen_convert()
                        self.antigen_model.load_state_dict(torch.load(antigen_state_path[2]))
                    cdr_pre = torch.tensor([AA_VOCAB['@']]*batch_size)
                    cdr_region = torch.tensor([4]*batch_size)
                else:
                    raise 
                cdr_pre = cdr_pre.reshape(-1,1)
                cdr_pre = torch.concat([cdr_pre,torch.zeros(cdr_pre.shape[0],29)],dim=1)
                cdr_pre = cdr_pre.long()
                ######################################################
                all_info = data.copy()
                all_info[3] = cdr_pre
                all_info[4] = torch.zeros(all_info[4].shape).long()
                if self.antigen:
                    antigen_esm = data[5]
                for i in range(1,29):
                    if self.sample:
                        if self.antigen:
                            
                            antigens = self.antigen_model(antigen_esm)
                            
                            data = generator(all_info,antigens)
                        else:
                            data = generator(all_info)
                        ####################
                        cc = []
                        for j in range(data.shape[0]):
                            cc.append(F.softmax(data[j][i-1],dim=-1).multinomial(1))
                        cc = torch.concat(cc)
                        ############################
                        # dist = Categorical(logits=data[:,i-1])
                        # cc = dist.sample()
                        
                        all_info[3][:,i] = cc # append a residue
                    else:
                        if self.antigen:
                            antigens = self.antigen_model(data[-1])
                            data = generator(all_info,antigens)
                        else:
                            data = generator(all_info)
                        cdr = torch.argmax(data,dim=-1).cpu()
                        all_info[3][:,i] = cdr[:,i-1] # append a residue
                    all_info[4][:,i] = cdr_region # append a residue region
                
                cdr = all_info[3]
                
                for i in range(cdr.shape[0]):
                    end_index = find_end(cdr[i])
                    cdrs.append(''.join([self.reverse_AA_VOVAB[int(x)] for x in cdr[i][1:end_index]]))
        return cdrs

    def generate(self,region,columns):
        region = region
        columns = columns
        for i in range(len(region)):
            
            pd.read_csv(self.gen_pos_path).dropna().to_csv(self.gen_pos_path,index=False) # drop nan cdr
            cdr = self.setup(region=region[i])
            cdr = pd.DataFrame(data=cdr,columns=[columns[i]])
        
            gen_pos = pd.read_csv(self.gen_pos_path)
            gen_pos[columns[i]] = cdr
            gen_pos = gen_pos.dropna()
            gen_pos.to_csv(self.gen_pos_path,index=False)
        print('done...')

    def validate_bind(self):
        config = configuration()
        


if __name__ == '__main__':
    cdr1_state_path = '../ckpts/pretrain/AntiTranslate50,pre_train,cdr1,6,6,512+6e-05.pth'
    cdr2_state_path = '../ckpts/pretrain/AntiTranslate50,pre_train,cdr2,6,6,512+6e-05.pth'
    cdr3_state_path = '../ckpts/pretrain/AntiTranslate50,pre_train,cdr3,6,6,512+6e-05.pth'
   # cdr3_state_path = '../ckpts/PPO/ACTOR_1_0.98.pth'
    antigen = True
    antigen_state_path = ['../ckpts/antigen_model.pth',
                          '../ckpts/antigen_model.pth',
                          '../ckpts/antigen_model.pth']
    
    gen_pos_path = '../data/example.csv'

    gpu = '0'

    staircase = starecase_generation(cdr1_state_path=cdr1_state_path,
                                     cdr2_state_path=cdr2_state_path,
                                     cdr3_state_path=cdr3_state_path,
                                     antigen_state_path=antigen_state_path,
                                     gpu=gpu,
                                     gen_pos_path=gen_pos_path,
                                     esm=False,
                                     sample=True,
                                     antigen=antigen)
    print(cdr1_state_path)
    print(gen_pos_path)
    region = ['cdr3']
    columns = ['H-CDR3']
    with torch.no_grad():
        staircase.generate(region=region,columns=columns)
