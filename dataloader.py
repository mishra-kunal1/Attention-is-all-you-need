from torch.utils.data import Dataset
import numpy as np
np.random.seed(42)
from tokenizer import CustomTokenizer
import torch
import json
from pickle import load
class TransLationDataloader(Dataset):
    def __init__(self,filepath,max_sentences):
        super().__init__()
        english_data,german_data=self.load_data_from_file(filepath)
        self.english_data=english_data[:max_sentences]
        self.german_data=german_data[:max_sentences]
        self.shuffle_indices=np.arange(len(self.english_data))
        np.random.shuffle(self.shuffle_indices)
        self.train_split=0.85
        self.test_split=0.15
        self.eng_train_data,self.ger_train_data,self.eng_test_data,self.ger_test_data=self.get_train_test_split()
       
        print("Train Data Size: ",len(self.eng_train_data))
        print("Test Data Size: ",len(self.eng_test_data))
        english_vocab=CustomTokenizer(dataset=self.eng_train_data)
        german_vocab=CustomTokenizer(dataset=self.ger_train_data)
        self.eng_tokenizer=english_vocab.get_vocab()
        self.ger_tokenizer=german_vocab.get_vocab()

        #get maximum length of german sentences
        # self.max_german_length=max([len(sentence.split()) for sentence in self.german_data])
        # self.max_english_length=max([len(sentence.split()) for sentence in self.english_data])
        self.max_german_length=33
        self.max_english_length=33
        print("Max German Sentence Length: ",self.max_german_length)
        print("Max English Sentence Length: ",self.max_english_length)
        #save eng and ger tokenizer
        with open('eng_tokenizer.json', 'w') as fp:
            json.dump(self.eng_tokenizer, fp)
        with open('ger_tokenizer.json', 'w') as fp:
            json.dump(self.ger_tokenizer, fp)
        print("English Vocab Size: ",len(self.eng_tokenizer))
        print("German Vocab Size: ",len(self.ger_tokenizer))
        
    
    def __len__(self):
        return len(self.english_data)
    def __getitem__(self,idx):
        return self.english_data[idx],self.german_data[idx]
    
    def load_data_from_file(self,filepath):
        dataset=load(open(filepath, 'rb'))
        english_sentences=['<sos>'+sentence[0]+'<eos>' for sentence in dataset]
        german_sentences=['<sos>'+sentence[1]+'<eos>' for sentence in dataset]
        return english_sentences,german_sentences
    
    
    def get_train_test_split(self):
        train_end_index=int(self.train_split*len(self.english_data))
        eng_train_data=self.english_data[:train_end_index]
        ger_train_data=self.german_data[:train_end_index]
        eng_test_data=self.english_data[train_end_index:]
        ger_test_data=self.german_data[train_end_index:]
        return eng_train_data,ger_train_data,eng_test_data,ger_test_data
    
    def create_encoded_data(self,data,vocab,max_len):
        encoded_data=torch.zeros((len(data),max_len),dtype=torch.long)
        for index,text in enumerate(data):

            encoded_vector=self.create_encoded_vector(text,vocab,max_len)
            encoded_data[index]=torch.tensor(encoded_vector,dtype=torch.long)
            
        return encoded_data
    
    def create_encoded_vector(self,text,vocab,max_len):
            text=text[5:-5]
            encoded_vector=[]
            encoded_vector.append(vocab['<sos>'])
            for i,token in enumerate(text.split()):
                if(i>=max_len-2):
                    break
                if token in vocab:
                    encoded_vector.append(vocab[token])

            encoded_vector.append(vocab['<eos>'])
            for k in range(max_len-len(encoded_vector)):
                encoded_vector.append(vocab['<pad>'])
            return encoded_vector

    
    def get_encoded_data(self):
        train_encoded_eng=self.create_encoded_data(self.eng_train_data,self.eng_tokenizer,self.max_english_length)
        train_encoded_ger=self.create_encoded_data(self.ger_train_data,self.ger_tokenizer,self.max_german_length)
        test_encoded_eng=self.create_encoded_data(self.eng_test_data,self.eng_tokenizer,self.max_english_length)
        test_encoded_ger=self.create_encoded_data(self.ger_test_data,self.ger_tokenizer,self.max_german_length)
        return train_encoded_eng,train_encoded_ger,test_encoded_eng,test_encoded_ger
            
        