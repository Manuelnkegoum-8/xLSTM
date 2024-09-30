import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler,mixed_prec, device,tokenizer,writer,epoch):
    avg_loss, avg_acc = 0.0, 0.0
    n = 0.0
    model.train()

    # Wrap dataloader with tqdm for a progress bar
    with tqdm(total=len(dataloader), desc="Training", colour='blue', ncols=100) as pbar:
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            texts, labels = data
            bs = texts.size(0)
            texts = texts.to(device)
            labels = labels.to(device)

            with autocast(enabled=mixed_prec):
                preds = model(texts)
                loss = criterion(preds, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            avg_loss += loss.item() * bs
            n += bs
            if i==0 and epoch%3==0:
                debug(preds, texts, labels, tokenizer,writer,epoch,num_samples=min(bs, 8),train=True)
            # Update the progress bar with the current loss
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    avg_loss /= n
    return avg_loss

@torch.no_grad()
def validate(model, dataloader, criterion, device, tokenizer,writer,epoch):
    avg_loss, avg_acc = 0.0, 0.0
    n = 0.0
    model.eval()

    # Wrap dataloader with tqdm for a progress bar
    with tqdm(total=len(dataloader), desc="Validation", colour='yellow', ncols=100) as pbar:
        for i, data in enumerate(dataloader):
            texts, labels = data
            bs = texts.size(0)
            texts = texts.to(device)
            labels = labels.to(device)

            preds = model(texts)
            loss = criterion(preds, labels)

            avg_loss += loss.item() * bs
            n += bs

            if i == 0:
                debug(preds, texts, labels, tokenizer,writer,epoch,num_samples=min(bs, 8),train=False)

            # Update the progress bar with the current loss
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

    avg_loss /= n
    return avg_loss

def debug(preds, texts, labels, tokenizer,writer,epoch, num_samples=4,train=True):
    preds = preds.argmax(dim=-1)  # Assuming classification task
    predictions = []
    targets = []
    semantics = []
    if train:
        tag_ = "DEBUG-TRAIN"
    else:
        tag_ = "DEBUG-VAL"
    for j in range(num_samples):
        input_text = tokenizer.decode(texts[j], skip_special_tokens=True)
        predicted_text = tokenizer.decode(preds[j], skip_special_tokens=True)
        true_text = tokenizer.decode(labels[j], skip_special_tokens=True)
        predictions.append(predicted_text)
        targets.append(true_text)
        semantics.append(input_text)
        """print(f"\nSample {j}:")
        logging.info(f"Input: {input_text}")
        logging.info(f"Prediction: {predicted_text}")
        logging.info(f"True Label: {true_text}")"""
    log_text = "  \n  \n".join([f"Input: {semantic}  \nPredicted output: {prediction}  \nExpected output: {target}\n"
                                        for semantic, prediction, target in zip(semantics, predictions, targets)])
    writer.add_text(tag=tag_,text_string=log_text,global_step = epoch)
        