import torch 
import torch.nn as nn
import torch.nn.functional as F 
from .mlstm import mLSTM_block

class xLSTMModel(nn.Module):
    def __init__(self,config=None):
        super(xLSTMModel,self).__init__()
        self.layers = nn.ModuleList()
        self.embedding_layer = nn.Embedding(config.vocab_size, config.embedding_dim)
        for i in range(config.m_layers):
            block = mLSTM_block(config.dim,config.heads)
            self.layers.append(block)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(config.dim,config.vocab_size)
    
    def forward(self, x):
        embedded = self.embedding_layer(x)
        for layer in self.layers:
            embedded = layer(embedded)
        out = self.fc(self.dropout(embedded[:, -1, :]))  # Use the last hidden state
        return out
    
    def decode(self,x):
        pass