import torch.nn as nn
import torch
device='mps' if torch.backends.mps.is_available() else 'cpu'
#device='cpu'
class TokenPositionEmbeddings(nn.Module):
    def __init__(self, vocab_size,max_len,embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len=max_len
        self.embedding_dim=embedding_dim
        #weight_word_embeddings=self.get_poistion_encoding(vocab_size,embedding_dim)
       
        self.token_embeddings = nn.Embedding(self.vocab_size, 
                                             self.embedding_dim,
                                             #_weight=weight_word_embeddings,
                                             #_freeze=True)
        )
        #embedding layer for position embeddings
        #it will use precomputed weights and will be frozen
        weight_position_embeddings=self.get_poistion_encoding(max_len,embedding_dim)
        self.position_embeddings = nn.Embedding(self.max_len, 
                                             self.embedding_dim,
                                             _weight=weight_position_embeddings,
                                             _freeze=True)
        
        
    def get_poistion_encoding(self,seq_length,hidden_size,n=10000):
        position_enc = torch.zeros(seq_length, hidden_size)
        for pos in range(seq_length):
            for i in range(hidden_size//2):
                
                position_enc[pos, 2*i] = torch.sin(torch.tensor(pos/(n**(2*i/hidden_size))))
                position_enc[pos, 2*i+1] = torch.cos(torch.tensor(pos/(n**((2*i+1)/hidden_size))))
        return position_enc
    
    def forward(self,inputs):
        #inputs is of shape [batch_size,max_len]
        batch_size,max_len=inputs.shape
        #create position indices
        position_indices=torch.arange(max_len).to(device)
        #get embeddings
        token_embeddings=self.token_embeddings(inputs)
        position_embeddings=self.position_embeddings(position_indices)
        #add both embeddings
        embeddings=token_embeddings+position_embeddings
        return embeddings