import torch
import torch.nn as nn
device='mps' if torch.backends.mps.is_available() else 'cpu'
#device='cpu'
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,query,key,value,mask=None):
        #print("Inside scaled dot product attention")
        B,nh,T,hs=query.shape
        key=key.transpose(-2,-1)
        #print('before calc attention score')
        #print('key shape',key.shape)
        #print('query shape',query.shape)
        attetntion_score=torch.matmul(query,key)
        #shape of attention_score is B,nh,T,T
        attetntion_score=attetntion_score/torch.sqrt(torch.tensor(hs))
        if(mask is not None):
            #print('mask',mask.shape)
            #print('attetntion_score',attetntion_score.shape)
            attetntion_score += (mask * -1e9)
        attention_weights=torch.softmax(attetntion_score,dim=-1)

        output=torch.matmul(attention_weights,value)
        #shape of output is B,nh,T,hs
        return output

       
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,embed_size,key_dim,query_dim,value_dim,mask=False):
        super().__init__()
        #print('inside multihead attention')
        self.num_heads=num_heads
        self.head_size=embed_size//num_heads
        self.key_dim=key_dim
        self.query_dim=query_dim
        self.value_dim=value_dim
        self.mask=mask
        self.attention=ScaledDotProductAttention()
        self.key_layer=nn.Linear(embed_size,key_dim).to(device)
        self.query_layer=nn.Linear(embed_size,query_dim).to(device)
        self.value_layer=nn.Linear(embed_size,value_dim).to(device)
        self.final_layer=nn.Linear(value_dim,embed_size).to(device)

    def forward(self,query,key,value,mask=None):
        #print('input shape',query.shape)
        B,T,C=query.shape
        
        query=self.query_layer(query)
        key=self.key_layer(key)
        value=self.value_layer(value)
        #print('query shape after dense layer',query.shape)
        #print('key shape after dense layer',key.shape)
        #print('value shape after dense layer',value.shape)
        query=query.view(B,T,self.num_heads,-1).transpose(1,2)
        #print('query shape after reshape',query.shape)
        key=key.view(B,T,self.num_heads,-1).transpose(1,2)
        value=value.view(B,T,self.num_heads,-1).transpose(1,2)
       

        scaled_attention=self.attention(query,key,value,mask)
        #shape of scaled_attention is B,nh,T,hs
        scaled_attention=scaled_attention.transpose(1,2).contiguous().view(B,T,-1)
        #shape of scaled_attention is B,T,nh*hs
        output=self.final_layer(scaled_attention)
        #print('output shape',output.shape)
        #shape of output is B,T,C
        return output

        

        
