import torch
from dataloader import TransLationDataloader
from transformer_model import TransformerModel
from pickle import load
import numpy as np
from torch.optim import Adam

import torch
def get_batch_data(batch_size,is_train=True):
    if is_train:
        eng=train_eng
        ger=train_ger
    else:
        eng=test_eng
        ger=test_ger
   #get random index as batch startinf index
    batch_start=np.random.randint(0,len(eng)-batch_size)
    batch_end_index=batch_start+batch_size
    batch_eng=eng[batch_start:batch_end_index]
    batch_ger=ger[batch_start:batch_end_index]
   
    return batch_eng,batch_ger

#optimizer

if __name__ == '__main__':
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    print('Using device', device)
    #device='cuda' if torch.cuda.is_available() else 'cpu'
    
    ger_max_len=33
    eng_max_len=33
    ger_vocab_size=8663
    eng_vocab_size=5627
    embedding_size=256
    num_blocks=6
    num_heads=8
    key_dim=64
    query_dim=64
    value_dim=64
    epochs=10000
    batch_size=32
    eval_iters=100
    
    dataset=TransLationDataloader('english-german_60k.pkl',15000)
    train_eng,train_ger,test_eng,test_ger=dataset.get_encoded_data()
    
    
    model=TransformerModel(src_max_length=eng_max_len,tar_max_length=ger_max_len,embedding_dim=embedding_size,key_dim=key_dim,query_dim=query_dim,
                            value_dim=value_dim,src_vocab_size=eng_vocab_size,tar_vocab_size=ger_vocab_size,dropout_rate=0.1,num_blocks=num_blocks,
                        num_heads=num_heads,device=device)
    #load weights
    #model.load_state_dict(torch.load('transformer_model.pt'))

    model=model.to(device)

    def rate(step, model_size, factor, warmup):
    
        if step == 0:
            step = 1
        return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
        )
    from torch.optim.lr_scheduler import LambdaLR
    optimizer = Adam(
            model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(
                step, embedding_size, factor=1, warmup=1000
            ),)
    
    @torch.no_grad()
    def estimate_loss():
        out = []
        model.eval()
        
        for split in [True, False]:
            
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                eng_batch,ger_batch=get_batch_data(batch_size,split)
                encoder_input=eng_batch[:,1:].to(device)
                decoder_input=ger_batch[:,:-1].to(device)
                decoder_target=ger_batch[:,1:].to(device)
                logits, loss = model(encoder_input, decoder_input,decoder_target)
                losses[k] = loss.item()
            out.append(losses.mean())
        model.train()
        return out
    
    for epoch in range(epochs):
        eng_batch,ger_batch=get_batch_data(batch_size,True)
        encoder_input=eng_batch[:,1:].to(device)
        decoder_input=ger_batch[:,:-1].to(device)
        decoder_target=ger_batch[:,1:].to(device)
        logits,loss=model(encoder_input,decoder_input,decoder_target)
        if(epoch%200==0):
           train_loss,test_loss=estimate_loss()
           print('Epoch: ',epoch,'Train Loss: ',train_loss.item(),'Test Loss: ',test_loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()


    

