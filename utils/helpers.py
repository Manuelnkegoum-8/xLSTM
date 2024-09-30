from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch
# Tokenize the dataset
def tokenize_function(examples,tokenizer,col):
    return tokenizer(examples[col],truncation=True)

def create_sequences(input_ids, context_length, padding_idx=0):
    in_seqs = []
    out_seqs = []
    
    # Use tqdm to display a progress bar
    for sequence in tqdm(input_ids, desc="Processing sequences", colour='green',ncols=100):
        if len(sequence) > context_length + 1:
            for i in range(len(sequence) - context_length):
                in_seqs.append(torch.LongTensor(sequence[i:i + context_length]))
                out_seqs.append(sequence[i + context_length])
    
    return pad_sequence(in_seqs, batch_first=True, padding_value=padding_idx), torch.LongTensor(out_seqs)


class ConfigXLSTM:
    def __init__(self, vocab_size, embedding_dim, m_layers, dim, heads, dropout_rate):
        self.vocab_size = vocab_size           # Vocabulary size
        self.embedding_dim = embedding_dim     # Embedding dimension
        self.m_layers = m_layers               # Number of LSTM layers
        self.dim = dim                         # Hidden dimension size
        self.heads = heads                     # Number of heads (for multi-head attention if needed)
        self.dropout_rate = dropout_rate       # Dropout rate for regularization


def generate(model,input_text,tokenizer,context,max_decode):
    inputs = tokenizer(input_text)['input_ids']
    for _ in range(max_decode):
        in_tensor = torch.LongTensor(inputs[-context:]).unsqueeze(0)
        pred = model(in_tensor) # 1*vocab
        token_pred = pred.argmax(dim=-1)[0]
        inputs.append(token_pred.item()) 
    return tokenizer.decode(inputs,skip_special_tokens=True)