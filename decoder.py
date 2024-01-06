import torch.nn as nn
import torch
from multihead_attention import MultiHeadAttention
from token_position_embeddings import TokenPositionEmbeddings
device='mps' if torch.backends.mps.is_available() else 'cpu'
#device='cpu'
class DecoderBlock(nn.Module):
    def __init__(self,num_heads,embed_size,key_dim,query_dim,value_dim):
        super().__init__()
        #print('inside decoder block')
        self.maskedMultiHeadAttention=MultiHeadAttention(num_heads,embed_size,key_dim,query_dim,value_dim)
        self.multihead_attention=MultiHeadAttention(num_heads,embed_size,key_dim,query_dim,value_dim)
        self.layer_norm=nn.LayerNorm(embed_size).to(device)
        self.feed_forward=nn.Sequential(
            nn.Linear(embed_size,embed_size*4),
            nn.ReLU(),
            nn.Linear(embed_size*4,embed_size)
        ).to(device)

    def forward(self,dec_input,enc_output,lookahead_mask,padding_mask):
        #print('inside decoder block')
        #print('enc_output',enc_output.shape)
        #print('dec_output',dec_input.shape)
        masked_attn_output=self.maskedMultiHeadAttention(dec_input,dec_input,dec_input,lookahead_mask)
        #print('masked_attn_output',masked_attn_output.shape)
        normalized_op1=self.layer_norm(dec_input+masked_attn_output)
        #print('normalized_op1',normalized_op1.shape)
        #print('enc_output',enc_output.shape)
        attn_output=self.multihead_attention(normalized_op1,enc_output,enc_output,padding_mask)
        #print('attn_output',attn_output.shape)
        normalized_op2=self.layer_norm(normalized_op1+attn_output)
        feed_forward_output=self.feed_forward(normalized_op2)
        normalized_op3=self.layer_norm(normalized_op2+feed_forward_output)
        return normalized_op3

class Decoder(nn.Module):
    def __init__(self,tar_max_length,embedding_dim,key_dim,value_dim,query_dim,tar_vocab_size,dropout_rate,num_blocks,num_heads,device='cpu'):
        super().__init__()
        #print('Initializing Decoder')
        self.max_length=tar_max_length
        self.token_position_embeddings=TokenPositionEmbeddings(tar_vocab_size,tar_max_length,embedding_dim)
        self.decoder_stack=[DecoderBlock(num_heads,embedding_dim,key_dim,value_dim,query_dim) for _ in range(num_blocks)]
        self.device=device
    
    def forward(self,inputs,enc_output,lookahead_mask,padding_mask):
        count=0
        x=self.token_position_embeddings(inputs)

        for decoder_block in self.decoder_stack:
            x=decoder_block(x,enc_output,lookahead_mask,padding_mask)
            #print('decoder Number',count)
            count+=1
        
        return x
