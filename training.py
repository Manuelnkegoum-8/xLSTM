import numpy as np
import random,json
import torch
from transformers import AutoTokenizer
import torch.nn as nn
import torch.optim
from models.xlstm import *
from utils.helpers import *
from utils.trainer import *
from torch.cuda.amp import GradScaler, autocast
from colorama import Fore, Style
import os
import torch.optim.lr_scheduler as lr_schedule
from utils.lr_scheduler import CosineWithLinearWarmup

import argparse
from tqdm import trange
from math import ceil
parser = argparse.ArgumentParser(description='Language Modelling')

# Data args
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--freq', default=10, type=int, metavar='N', help='log frequency (by iteration)')
parser.add_argument('--context', default=15, type=int, help='context lenght')


# Model parameters

parser.add_argument('--heads', default=4, type=int, help='number of heads')
parser.add_argument('--m_blocks', default=4, type=int, help='number of mlstm blocks')
parser.add_argument('--s_blocks', default=4, type=int, help='number of slstm blocks')
parser.add_argument('--dim', default=512, type=int, help='embedding dim of patch')
parser.add_argument('--dropout', default=0.2, type=float, help='embedding dim of patch')

# Optimization hyperparams
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--amp', default=False, type=bool, metavar='N', help='mixed precision')

parser.add_argument('--warmup', default=5, type=int, metavar='N', help='number of warmup epochs')
parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', default=0.1, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default=False, help='Version')

args = parser.parse_args()
lr = args.lr
weight_decay = args.weight_decay
dim, heads = args.dim, args.heads
m_blocks = args.m_blocks
s_blocks = args.s_blocks
dropout = args.dropout
batch_size = args.batch_size
warmup = args.warmup

device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


if __name__ == '__main__':
    
    
    # Apply tokenization
    # Initialize the tokenizer (BERT in this case, but can be GPT-2 or others)
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')
    # Load WikiText-2 dataset
    dataset = load_dataset("FareedKhan/1k_stories_100_genre")

    train_data = dataset['train'].select(range(10)).map(tokenize_function, batched=True,fn_kwargs={'tokenizer': tokenizer, 'col': 'story'})
    train_data = train_data['input_ids']
    #val_data = dataset['validation'].select(range(10)).map(tokenize_function, batched=True,fn_kwargs={'tokenizer': tokenizer, 'col': 'story'})
    val_data = train_data #val_data['input_ids']
    # Define the context length (K previous tokens)
    context_length = args.context
    inputs, targets = create_sequences(train_data, context_length)
    val_inputs,val_targets = create_sequences(val_data, context_length)
    train_dataset = TensorDataset(inputs, targets)
    val_dataset = TensorDataset(val_inputs, val_targets)

    # Create TensorDataset and DataLoader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    

    config = ConfigXLSTM(vocab_size = tokenizer.vocab_size, 
                         embedding_dim = dim ,
                         m_layers = m_blocks , 
                         dim = dim , 
                         heads = heads , 
                         dropout_rate = dropout)
    model = xLSTMModel(config)
    total_params = sum(p.numel() for p in model.parameters())

    # Print the number of parameters
    print(f"Number of parameters: {total_params}")
    model = model.to(device)
    optim_groups = model.no_weight_decay()

    optimizer = torch.optim.AdamW(
        (
            {"weight_decay": weight_decay, "params": optim_groups[1]},
            {"weight_decay": 0.0, "params": optim_groups[0]},
        ),
        lr=lr,
    )
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss().to(device)
    num_epochs = args.epochs
    min_steps = ceil(len(train_dataset)/batch_size)
    scheduler = CosineWithLinearWarmup(optimizer, warmup_steps=4000, total_steps=int(args.epochs * min_steps),max_lr=lr)

    # Train the model
    best_loss = float('inf')
    torch.autograd.set_detect_anomaly(True)
    start_epoch = -1
    final_epoch = args.epochs
    if args.resume:
        checkpoint = torch.load('ckpt_lm.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
    
    print(Fore.LIGHTGREEN_EX+'='*100)
    print("[INFO] Begin training for {0} epochs".format(final_epoch-start_epoch+1))
    print('='*100+Style.RESET_ALL)
    # Initialize TensorBoard writer
    writer = SummaryWriter('m_lstm_xp')
    for epoch in range(start_epoch+1,final_epoch):
        train_loss = train_epoch(model,train_loader,optimizer,scheduler,criterion,scaler,args.amp,device,tokenizer,writer,epoch)
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        torch.cuda.empty_cache()
        if epoch%args.freq==0 or epoch==args.epochs-1:
            valid_loss = validate(model,val_loader,criterion,device,tokenizer,writer,epoch)
            writer.add_scalar('Loss/Validation', valid_loss, epoch)
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {valid_loss}")
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }, f"ckpt_lm.pt")
    writer.close()
    print(Fore.GREEN+'='*100)
    print("[INFO] End training")
    print('='*100+Style.RESET_ALL)