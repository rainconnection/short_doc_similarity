import pandas as pd
import numpy as np
import re

import torch
import torchvision
from torch import tensor, nn
from torch.utils.data import TensorDataset, DataLoader

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def sub_special_characters(text):
    pattern = r'[^a-zA-Z0-9\s]'  # 알파벳, 숫자, 공백 문자가 아닌 모든 문자
    without_special_chars = re.sub(pattern, ' ', text)
    #without_special_chars = without_special_chars.lower()
    return ' '.join(without_special_chars.split()).strip()

### pre_data
def padding(data):
    sent_len = max([len(sent) for sent in data])
    padded = np.zeros((len(data),sent_len)).astype("float32")
    mask = np.zeros((len(data),sent_len)).astype("float32")

    for idx, sent in enumerate(data):
        padded[idx, :len(sent)] = np.asarray(sent).astype("float32")
        mask[idx, :len(sent)] = 1.0
    
    return padded, mask


def tokenizing(docs, max_len, tokenizer) : 
    # load bert tokenizer
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)

    token_ids = []
    for sents in docs:
        tokenized_sents = [tokenizer.tokenize(sent) for sent in sents]
    
        add_special_sents = ["[CLS]"]
        for sent in tokenized_sents:
            add_special_sents += sent[:max_len - 2]
            add_special_sents += ["[SEP]"]
        add_special_sents
        token_ids.append(tokenizer.convert_tokens_to_ids(add_special_sents))
    return token_ids


def preprocess(docs, tokenizer, M_len = 256):
    
    # sentence clean-up
    docs = [[sub_special_characters(sent) for sent in sents] for sents in docs]
    
    # token
    token_idx_docs = tokenizing(docs, M_len, tokenizer = tokenizer)
    # Pad Sequence
    sent, mask = padding(token_idx_docs)
    #print(test_text_inputs[0])

    sent = tensor(sent).long().to(device)
    mask = tensor(mask).to(device)
    
    return sent, mask #encoding(sent, mask)

### pre_modeling

def set_dataLoader(*datas, **kwargs):
    #label = label.reshape(label.shape[0],1)
    if 'batch_size' in kwargs:
        batch_size = kwargs['batch_size']
    else:
        batch_size = 96

    if 'type' in kwargs:
        type = kwargs['type']
    else:
        type = 'train'
        
    dataloader = TensorDataset(*datas)
    
    if type == 'valid':
        dataloader = DataLoader(dataloader, shuffle = False, batch_size = batch_size)
    elif type == 'train':
        dataloader = DataLoader(dataloader, shuffle = True, batch_size = batch_size)
    elif type == 'test':
        dataloader = DataLoader(dataloader, shuffle = False, batch_size = batch_size)
    else:
        print("Please input 'type'")
        return None
        
    return dataloader


### for modeling
def get_infer(ids1, mask1, ids2, mask2):
    # bert-encoding
    layer1 = encoding(ids1, mask1)
    layer2 = encoding(ids2, mask2)
    
    # siamese
    last_layer, _softmax = model(layer1, layer2)
    
    return last_layer, _softmax