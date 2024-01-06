import torchtext
import torch
class CustomTokenizer:
    def __init__(self, dataset=None,vocab=None):
        # Create a custom vocabulary from your dataset
        self.eng_75=8
        self.ger_75=33

        if(vocab==None):
            self.vocab = {}
            self.pad_index = 0
            self.vocab['<pad>'] = self.pad_index
            self.sos_index = len(self.vocab)
            self.vocab['<sos>'] = self.sos_index
            self.eos_index = len(self.vocab)
            self.vocab['<eos>'] = self.eos_index

            for text in dataset: 
                text=text[5:-5]    
                for token in text.split():
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)  
        else:
            self.vocab=vocab

    def get_vocab(self):
        return self.vocab
    

    def tokenize(self, text,max_length):
        # Tokenize a text using the custom vocabulary
        text=text[5:-5]
        encoded_vector=[]
        encoded_vector.append(self.vocab['<sos>'])
        text = ''.join([i for i in text if i.isalnum() or i == ' '])
        for i,token in enumerate(text.split()):
            if(i>=max_length-2):
                break
            if token in self.vocab:
                encoded_vector.append(self.vocab[token])

        encoded_vector.append(self.vocab['<eos>'])
        if(len(encoded_vector)<max_length):
            #add padding till max_length
            for i in range(max_length-len(encoded_vector)):
                encoded_vector.append(self.vocab['<pad>'])
        
        return torch.tensor(encoded_vector)
    
    def tokenize_batch(self, text_list,max_len):
        # Tokenize a list of texts using the custom vocabulary
        encoded_batch=torch.stack([self.tokenize(text,max_len) for text in text_list])
        return encoded_batch

    def get_vocab_size(self):
        return len(self.vocab)