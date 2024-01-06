import torch.nn as nn
import torch
from multihead_attention import MultiHeadAttention
from token_position_embeddings import TokenPositionEmbeddings
device='mps' if torch.backends.mps.is_available() else 'cpu'

class EncoderBlock(nn.Module):
    def __init__(self,num_heads,embed_size,key_dim,query_dim,value_dim):
        super().__init__()
        
        self.MultiHeadAttention=MultiHeadAttention(num_heads,embed_size,key_dim,query_dim,value_dim)
        self.layer_norm=nn.LayerNorm(embed_size).to(device)
        self.feed_forward=nn.Sequential(
            nn.Linear(embed_size,4*embed_size),
            nn.ReLU(),
            nn.Linear(4*embed_size,embed_size)
        ).to(device)
    
    def forward(self,inputs,mask):
        attention=self.MultiHeadAttention(inputs,inputs,inputs,mask)

        normalized_op1=self.layer_norm(inputs+attention)
        feed_forward_op=self.feed_forward(normalized_op1)
        normalized_op2=self.layer_norm(normalized_op1+feed_forward_op)
        return normalized_op2


class Encoder(nn.Module):
    def __init__(self,src_max_length,embedding_dim,key_dim,query_dim,value_dim,src_vocab_size,dropout_rate,num_blocks,num_heads):
        super().__init__()
        #print('Initializing Encoder')
        self.max_length=src_max_length
        self.token_position_embeddings=TokenPositionEmbeddings(src_vocab_size,src_max_length,embedding_dim)
        #self.dropout=nn.Dropout(dropout_rate)
        #head_size=embedding_dim//num_heads
        self.encoder_stack=[EncoderBlock(num_heads,embedding_dim,key_dim,query_dim,value_dim) for _ in range(num_blocks)]
    
    def forward(self,inputs,mask):
        #print('inside encoder')
        #print('inputs shape',inputs.shape)
        #print('mask shape',mask.shape)
        x=self.token_position_embeddings(inputs)

        #print('position embeddings  shape',x.shape)
        count=0
        for encoder_block in self.encoder_stack:
            x=encoder_block(x,mask)
            #print('encoder Number',count)
            count+=1
        
        #print('encoder output shape',x.shape)
        return x

        
