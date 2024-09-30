import torch 
import torch.nn as nn
import torch.nn.functional as F 
from .mlstm import mLSTM_block

class xLSTMBlock(nn.Module):
    def __init__(self,dim,heads,m_layers,s_layers):
        super(xLSTMBlock,self).__init__()
        self.layers = nn.ModuleList()
        for i in range(m_layers):
            block = mLSTM_block(dim,heads)
            self.layers.append(block)
    
    def forward(self, x):
        embedded = x.clone()
        for layer in self.layers:
            embedded = layer(embedded)
        out = embedded + x
        return out
    
    def forward_step(self,x,state):
        embedded = x.clone()
        for layer in layers:
            embedded,state = layer.step(x,state)
        out = embedded + x
        return out,state


class xLSTMModel(nn.Module):
    def __init__(self,config=None):
        super(xLSTMModel,self).__init__()
        self.block = xLSTMBlock(config.dim,config.heads,config.m_layers,None)
        self.embedding_layer = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(config.dim,config.vocab_size)
    
    def forward(self, x):
        embedded = self.embedding_layer(x)
        embedded = self.block(embedded)
        out = self.fc(self.dropout(embedded[:, -1, :]))  # Use the last hidden state
        return out     


    def no_weight_decay(self):
            no_decay = [p for n, p in self.named_parameters() if 'embedding' in n]
            decay = [p for n, p in self.named_parameters() if 'embedding' not in n]
            return no_decay, decay