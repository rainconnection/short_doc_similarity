#import torchvision.models as models
from torch import tensor
from torch.optim import AdamW

import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision

import numpy as np

#from transformers import BertTokenizer, BertModel, ElectraModel, ElectraTokenizer

class embeding_model(nn.Module):
    # bert기반 모형의 pooling-model
    # cls, mean, max pooling 가능
    # mean pooling을 기본으로 함
    def __init__(self, model, pooling_type = 'mean'):
        super().__init__()
        self.encoder = model # fretrained_embedding model load
        self.pooling_type = pooling_type
        
    def forward(self, ids, mask):
        embedded = self.encoder(ids, attention_mask=mask)
        attention_mask = mask
        last_hidden_layer = embedded.last_hidden_state
        pooling_type = self.pooling_type
        if pooling_type == 'cls':
            pooled = embedded.pooler_output
            
        elif pooling_type == 'mean':
            mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_layer.size()).float()
            )

            embedded_sum = torch.sum(last_hidden_layer * mask_expanded, 1)
    
            mask_sum = mask_expanded.sum(1)
            mask_sum = torch.clamp(mask_sum, min=1e-9)
    
            pooled = embedded_sum/mask_sum
            
        elif pooling_type == 'max':
            mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )

            last_hidden_layer[mask_expanded == 0] = -1e9
            max_over_time = torch.max(last_hidden_layer,1)[0]

            pooled = max_over_time
            
        else:
            print('please enter pooling_type [cls, mean, max]')
            return None
            
        return pooled
        



class doc_sim_model(nn.Module):
    def __init__(self, embedding_model, add_layers = None, initialize = None):
        super().__init__()
        self.type = 'cos'
        self.embedding_model = embedding_model
        
        # layer 추가 가능
        if add_layers is not None:
            self.layers = add_layers
            # add layer에 대한 initialize
            if initialize is not None:
                for i,x in enumerate(self.layers.modules()):
                    if isinstance(x, nn.Linear):
                        initialize(x.weight.data)
                        print('initial_lize : ', x)
        else:
            self.layers = None
    def forward(self, ids1, mask1, ids2, mask2):
        emb1 = self.embedding_model(ids1, mask1)
        emb2 = self.embedding_model(ids2, mask2)
        
        if self.layers is not None:
            emb1 = self.layers(emb1)
            emb2 = self.layers(emb2)
        
        # doc 1, doc 2에 대한 Cosine Similarity 계산
        cos_score_transformation = nn.Identity()
        outputs = cos_score_transformation(torch.cosine_similarity(emb1, emb2))

        return outputs

