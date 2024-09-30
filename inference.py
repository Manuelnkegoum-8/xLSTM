import torch
from utils.helpers import generate,ConfigXLSTM
from models.xlstm import xLSTMModel
import os
from transformers import AutoTokenizer
import argparse
import gradio as gr

parser = argparse.ArgumentParser(description='Language Modelling')
parser.add_argument('--context', default=15, type=int, help='context lenght')
parser.add_argument('--max_decode', default=15, type=int, help='max decode lenght')

# Model parameters
parser.add_argument('--heads', default=4, type=int, help='number of heads')
parser.add_argument('--m_blocks', default=4, type=int, help='number of mlstm blocks')
parser.add_argument('--s_blocks', default=4, type=int, help='number of slstm blocks')
parser.add_argument('--dim', default=512, type=int, help='embedding dim of patch')
parser.add_argument('--dropout', default=0.2, type=float, help='embedding dim of patch')


args = parser.parse_args()



dim, heads = args.dim, args.heads
m_blocks = args.m_blocks
s_blocks = args.s_blocks
dropout = args.dropout
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
config = ConfigXLSTM(vocab_size = tokenizer.vocab_size, 
                            embedding_dim = dim ,
                            m_layers = m_blocks , 
                            dim = dim , 
                            heads = heads , 
                            dropout_rate = dropout)
model = xLSTMModel(config)
try:
        checkpoint = torch.load('ckpt_lm.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        print('[INFO] Succesfully loaded checkpoint')
except:
        print('[INFO] No pretrained checkpoint available or compatible for this config, model loaded with random weights')
model = model.to(device)

def predict(text):
    out = generate( model=model,
                    input_text=text,
                    tokenizer=tokenizer,
                    context=args.context,
                    max_decode=args.max_decode)
    return out
if __name__ == '__main__':
        
    demo = gr.Interface(
    fn=predict,
    inputs=["text"],
    outputs=["text"],
    ) 
    demo.launch()

