from encoder import Encoder
from decoder import Decoder
import torch.nn as nn
import torch
import numpy as np
#import cross entropy loss
from torch.nn import CrossEntropyLoss
device='mps' if torch.backends.mps.is_available() else 'cpu'
#device='cpu'
class TransformerModel(nn.Module):
    def __init__(self,src_max_length,tar_max_length,embedding_dim,key_dim,query_dim,value_dim,src_vocab_size,tar_vocab_size,dropout_rate,num_blocks,num_heads,device='cpu'):
        super().__init__()
        #print('Initializing Transformer Model')
        self.encoder=Encoder(src_max_length,embedding_dim,key_dim,query_dim,value_dim,src_vocab_size,dropout_rate,num_blocks,num_heads)
        self.decoder=Decoder(tar_max_length,embedding_dim,key_dim,query_dim,value_dim,tar_vocab_size,dropout_rate,num_blocks,num_heads)
        self.final_layer=nn.Linear(embedding_dim,tar_vocab_size)
    
    def create_padding_mask(self,inputs):
        mask=torch.zeros(inputs.shape[0],inputs.shape[1]).to(device)
        mask=mask.masked_fill(inputs==0,1)
        mask=mask.view(inputs.shape[0],1,1,inputs.shape[1])
        return  mask
    
    def create_lookahead_mask(self,inputs):
        mask=torch.triu(torch.ones((inputs.shape[1],inputs.shape[1])),diagonal=1)
        return mask
    
    def forward(self,enc_inputs,dec_inputs,target):
    
        padding_mask_enc=self.create_padding_mask(enc_inputs)
        padding_mask_dec=self.create_padding_mask(dec_inputs)
        lookahead_mask_dec=self.create_lookahead_mask(dec_inputs).to(device)
        #print('padding_mask_dec',padding_mask_dec.shape)
        #print('lookahead_mask_dec',lookahead_mask_dec.shape)
        dec_mask= torch.max(padding_mask_dec,lookahead_mask_dec)

        #padding_mask_enc=padding_mask_enc.to(device)
        #dec_mask=dec_mask.to(device)
 
        enc_output=self.encoder(enc_inputs,padding_mask_enc)
        #print('Encoder working')
        dec_output=self.decoder(dec_inputs,enc_output,dec_mask,padding_mask_dec)
        #print('Decoder working')
        output=self.final_layer(dec_output)
        loss=None
        if(target is not None):
            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(output.view(-1, output.shape[-1]), target.view(-1))


        return output,loss